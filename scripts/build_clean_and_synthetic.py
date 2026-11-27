"""MicrobeSim: clean the real ComBase observations, enrich them, and top each organism
up to the 4,000-observation target with clearly-labelled, real-data-calibrated
synthetic curves.

Run from anywhere:
    python build_clean_and_synthetic.py --raw microbesim_real_observations.csv \
        --rates-xlsx microbesim_real_dataset.xlsx --project-root /path/to/microbesim \
        --out out_dir

Deterministic (fixed seed). Real rows are never altered in value; every synthetic row
is flagged (is_synthetic=True) and points at the real curve it was modelled on.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LN10 = float(np.log(10.0))
TARGET_PER_ORGANISM = 4000
SEED = 20261031
SENTINEL = -0.01  # ComBase "not detected / below detection limit" marker in the source file

ORG_MAP = {
    "Bacillus cereus": "Bacillus cereus",
    "Campylobacteraceae (c.jejuni and relatives)": "Campylobacter jejuni",
    "Clostridium botulinum (non-prot.)": "Clostridium botulinum (non-proteolytic)",
    "Clostridium botulinum (prot.)": "Clostridium botulinum (proteolytic)",
    "Clostridium perfringens": "Clostridium perfringens",
    "Escherichia coli": "Escherichia coli",
    "Listeria monocytogenes/innocua": "Listeria monocytogenes",
    "Salmonella spp": "Salmonella spp.",
    "Staphylococcus aureus": "Staphylococcus aureus",
    "Yersinia enterocolitica": "Yersinia enterocolitica",
}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


# =============================================================================
# 1. CLEANING
# =============================================================================
def clean_real(raw: pd.DataFrame, log: list) -> pd.DataFrame:
    df = raw.copy()
    n0 = len(df)
    log.append(("input rows", n0, "microbesim_real_observations.csv"))

    # --- exact duplicates (already de-duplicated upstream; verify) -----------------
    key = [
        "organism_group",
        "record_id",
        "time_h",
        "population_log10_cfu_source_value",
        "temperature_c",
        "ph",
        "water_activity_aw",
        "food_name",
    ]
    dups = int(df.duplicated(key).sum())
    df = df.drop_duplicates(key).reset_index(drop=True)
    log.append(("exact duplicate rows removed", dups, "organism+record+time+count+conditions"))

    # --- harmonise names ------------------------------------------------------------
    df["organism_source_label"] = df["organism_group"]
    df["organism"] = df["organism_group"].map(ORG_MAP)
    assert df["organism"].notna().all(), "unmapped organism label"
    log.append(
        (
            "organism labels harmonised to project names",
            int((df.organism != df.organism_source_label).sum()),
            "e.g. 'Salmonella spp' -> 'Salmonella spp.'",
        )
    )
    fn = df["food_name"].str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
    log.append(
        (
            "food_name case/whitespace variants merged",
            int(df.food_name.nunique() - fn.nunique()),
            f"{df.food_name.nunique()} -> {fn.nunique()} distinct names",
        )
    )
    df["food_name"] = fn

    # --- sentinel counts -------------------------------------------------------------
    src = df["population_log10_cfu_source_value"].astype(float)
    df["log10_count_source_value"] = src
    df["below_detection_limit"] = src < 0
    df["log10_count"] = src.where(~df["below_detection_limit"])
    log.append(
        (
            "'-0.01' sentinel counts -> NaN + below_detection_limit flag",
            int(df.below_detection_limit.sum()),
            "kept as rows so the time design is preserved; excluded from fitting",
        )
    )

    # --- implausible water activity --------------------------------------------------
    df["temperature_celsius"] = df["temperature_c"]
    df["ph"] = df["ph"]
    df["water_activity"] = df["water_activity_aw"]
    bad_aw = df["water_activity"] < 0.5
    df["aw_implausible_removed"] = bad_aw
    df.loc[bad_aw, "water_activity"] = np.nan
    log.append(
        (
            "aw < 0.5 set to NaN (flagged)",
            int(bad_aw.sum()),
            "record O091_1/O091_2 (biscuit, aw=0.08): impossible for a growth/inactivation matrix",
        )
    )
    bad_ph = (df["ph"] < 2) | (df["ph"] > 11)
    df.loc[bad_ph, "ph"] = np.nan
    log.append(("pH outside [2, 11] set to NaN", int(bad_ph.sum()), ""))

    # --- structured fields parsed out of free text ----------------------------------
    notes = df["conditions_notes"].fillna("")
    df["nacl_pct"] = pd.to_numeric(
        notes.str.extract(r"Sodium chloride\s+in the environment:\s*([\d.]+)\s*\(%\)")[0],
        errors="coerce",
    )
    df["note_cut_minced"] = notes.str.contains("Cut (minced", regex=False)
    df["note_raw"] = notes.str.contains(r"\bRaw\b", regex=True)
    df["note_shaken"] = notes.str.contains("Shaken", regex=False)
    df["note_mixed_inoculation"] = notes.str.contains(
        "Other species also deliberately inoculated", regex=False
    )
    asm = df["assumed_conditions"].fillna("").str.lower()
    df["assumed_ph"] = asm.str.contains(r"\bph\b", regex=True)
    df["assumed_aw"] = asm.str.contains(r"\baw\b", regex=True)
    df["assumed_temperature"] = asm.str.contains("temp", regex=False)
    df["assumed_initial_count"] = asm.str.contains("logc0", regex=False)
    log.append(
        (
            "NaCl % parsed from conditions_notes",
            int(df.nacl_pct.notna().sum()),
            "new column nacl_pct",
        )
    )

    # --- tidy ordering / ids --------------------------------------------------------
    df["time_hours"] = df["time_h"].astype(float)
    df["curve_id"] = df["organism"].map(slug) + "__" + df["record_id"].astype(str)
    df["data_origin"] = "real_combase"
    df["is_synthetic"] = False
    df["synthetic_donor_curve_id"] = pd.NA
    df = df.sort_values(["curve_id", "time_hours", "observation_id"]).reset_index(drop=True)
    return df


# =============================================================================
# 2. DERIVED COLUMNS
# =============================================================================
def add_bands(df: pd.DataFrame) -> pd.DataFrame:
    T = df["temperature_celsius"]
    df["temperature_band"] = pd.cut(
        T,
        [-np.inf, 0, 8, 25, 45, np.inf],
        labels=[
            "frozen (<=0C)",
            "refrigeration (0-8C)",
            "abuse/cool (8-25C)",
            "ambient/warm (25-45C)",
            "thermal (>45C)",
        ],
    ).astype("object")
    df["ph_class"] = np.where(
        df["ph"].isna(), None, np.where(df["ph"] < 4.6, "acidic (<4.6)", "low-acid (>=4.6)")
    )
    aw = df["water_activity"]
    df["aw_class"] = pd.cut(
        aw,
        [0, 0.85, 0.92, 1.0001],
        labels=["low (<0.85)", "intermediate (0.85-0.92)", "high (>0.92)"],
    ).astype("object")
    df["metadata_complete"] = (
        df[["temperature_celsius", "ph", "water_activity"]].notna().all(axis=1)
    )
    return df


def _curve_stats(g: pd.DataFrame) -> dict:
    g = g.sort_values("time_hours")
    t_all = g["time_hours"].to_numpy()
    ok = g["log10_count"].notna().to_numpy()
    t, y = t_all[ok], g["log10_count"].to_numpy()[ok]
    out = dict(
        n_rows=len(g),
        n_points=int(ok.sum()),
        n_below_detection=int((~ok).sum()),
        n_unique_times=int(len(np.unique(t))) if len(t) else 0,
        duration_h=float(t_all.max() - t_all.min()),
    )
    if len(t) == 0:
        out.update(behavior="no_valid_counts", y_initial=np.nan)
        return out
    t0 = t.min()
    y0 = float(np.mean(y[t == t0]))
    t1 = t.max()
    y_last = float(np.mean(y[t == t1]))
    rise, fall = float(y.max() - y0), float(y0 - y.min())
    out.update(
        y_initial=y0,
        y_last=y_last,
        y_peak=float(y.max()),
        y_min=float(y.min()),
        net_change_log10=y_last - y0,
        peak_rise_log10=rise,
        max_fall_log10=fall,
    )
    if out["n_unique_times"] < 2:
        beh = "single_time"
    elif rise > 0.2 and rise >= fall:
        beh = "growth"
    elif fall > 0.2:
        beh = "decline"
    else:
        beh = "flat"
    out["behavior"] = beh
    if out["n_unique_times"] >= 2 and t.max() > t.min():
        sl = stats.linregress(t, y)
        out["slope_log10_per_h"] = float(sl.slope)
    return out


def fit_curves(obs: pd.DataFrame, primary) -> pd.DataFrame:
    """One row per curve: behaviour class + Gompertz fit (same acceptance rules as src/combase.py)."""
    fit_primary_curve, FitError = primary
    rows = []
    for cid, g in obs.groupby("curve_id", sort=False):
        s = _curve_stats(g)
        s["curve_id"] = cid
        s.update(fit_ok=False, fit_reject_reason=None)
        if s["behavior"] == "growth":
            gg = g.dropna(subset=["log10_count"]).sort_values("time_hours")
            if len(gg) < 6:
                s["fit_reject_reason"] = "fewer than 6 points"
            else:
                t = gg["time_hours"].to_numpy()
                y = LN10 * (gg["log10_count"].to_numpy() - gg["log10_count"].to_numpy()[:3].min())
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        f = fit_primary_curve(t, y)
                    rel = f.param_std[1] / f.mu_max if f.mu_max > 0 else np.inf
                    s.update(
                        fit_mu_max_per_h=f.mu_max,
                        fit_lag_h=f.lam,
                        fit_increase_log10=f.A / LN10,
                        fit_r2=f.r_squared,
                        fit_rmse_ln=f.rmse,
                        fit_mu_rel_se=rel,
                        fit_doubling_time_h=f.doubling_time_h,
                    )
                    if f.mu_max <= 1.5e-3:
                        s["fit_reject_reason"] = "mu_max at optimizer floor"
                    elif f.r_squared < 0.8:
                        s["fit_reject_reason"] = "R2 < 0.8"
                    elif not np.isfinite(rel) or rel > 0.5:
                        s["fit_reject_reason"] = "relative SE of mu_max > 50%"
                    else:
                        s["fit_ok"] = True
                except FitError as exc:
                    s["fit_reject_reason"] = f"fit failed: {str(exc)[:60]}"
        elif s["behavior"] in ("decline", "flat", "single_time", "no_valid_counts"):
            s["fit_reject_reason"] = f"not a growth curve ({s['behavior']})"
        rows.append(s)
    cs = pd.DataFrame(rows)
    meta_cols = [
        "curve_id",
        "record_id",
        "organism",
        "food_category",
        "food_name",
        "temperature_celsius",
        "ph",
        "water_activity",
        "nacl_pct",
        "assumed_ph",
        "assumed_aw",
        "assumed_temperature",
        "assumed_initial_count",
        "data_origin",
        "is_synthetic",
        "synthetic_donor_curve_id",
    ]
    meta = obs.drop_duplicates("curve_id")[meta_cols]
    cs = meta.merge(cs, on="curve_id", how="right")
    cs["metadata_complete"] = (
        cs[["temperature_celsius", "ph", "water_activity"]].notna().all(axis=1)
    )
    cs["usable_for_secondary_model"] = cs["fit_ok"] & cs["metadata_complete"]
    return cs


def assign_split(cs: pd.DataFrame, rng) -> pd.Series:
    """70/15/15 split by CURVE inside each organism (curves, not points, are the independent unit)."""
    split = pd.Series("train", index=cs.index)
    real = cs[~cs["is_synthetic"]]
    for _, idx in real.groupby("organism").groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        n = len(idx)
        split.loc[idx[int(0.70 * n) : int(0.85 * n)]] = "validation"
        split.loc[idx[int(0.85 * n) :]] = "test"
    return split


def attach_curve_columns(obs: pd.DataFrame, cs: pd.DataFrame) -> pd.DataFrame:
    keep = cs[
        [
            "curve_id",
            "behavior",
            "n_points",
            "duration_h",
            "y_initial",
            "fit_mu_max_per_h",
            "fit_lag_h",
            "fit_r2",
            "fit_ok",
            "slope_log10_per_h",
            "usable_for_secondary_model",
            "split",
        ]
    ].rename(
        columns={
            "behavior": "curve_behavior",
            "n_points": "curve_n_points",
            "duration_h": "curve_duration_h",
            "y_initial": "curve_log10_initial",
            "fit_mu_max_per_h": "curve_mu_max_per_h",
            "fit_lag_h": "curve_lag_h",
            "fit_r2": "curve_fit_r2",
            "fit_ok": "curve_growth_fit_ok",
            "slope_log10_per_h": "curve_slope_log10_per_h",
            "usable_for_secondary_model": "usable_for_secondary_model",
        }
    )
    obs = obs.drop(
        columns=[c for c in keep.columns if c in obs.columns and c != "curve_id"], errors="ignore"
    )
    obs = obs.merge(keep, on="curve_id", how="left")
    obs["log10_change_from_initial"] = obs["log10_count"] - obs["curve_log10_initial"]
    obs["ln_change_from_initial"] = obs["log10_change_from_initial"] * LN10
    obs["usable_for_primary_fit"] = (
        obs["curve_growth_fit_ok"].fillna(False) & obs["log10_count"].notna()
    )

    flags = []
    for col, cond, name in [
        ("below_detection_limit", obs["below_detection_limit"], "below_detection_sentinel"),
        ("aw", obs["aw_implausible_removed"].fillna(False), "aw_implausible_removed"),
        ("hi", obs["log10_count"] > 10, "count_gt_10_log"),
        ("ph_missing", obs["ph"].isna(), "ph_missing"),
        (
            "aw_missing",
            obs["water_activity"].isna() & ~obs["aw_implausible_removed"].fillna(False),
            "aw_missing",
        ),
        ("t_missing", obs["temperature_celsius"].isna(), "temperature_missing"),
        (
            "assumed",
            obs["assumed_ph"].fillna(False)
            | obs["assumed_aw"].fillna(False)
            | obs["assumed_temperature"].fillna(False),
            "condition_assumed_by_combase",
        ),
        ("single", obs["curve_behavior"].eq("single_time"), "single_time_point_curve"),
    ]:
        flags.append(np.where(cond, name + ";", ""))
    obs["qc_flags"] = ["".join(x).rstrip(";") or "ok" for x in zip(*flags)]
    obs["include_for_modeling"] = (
        obs["log10_count"].notna()
        & obs["metadata_complete"]
        & ~obs["curve_behavior"].isin(["single_time", "no_valid_counts"])
    )
    return obs


# =============================================================================
# 3. SYNTHETIC GENERATION (calibrated on the real curves)
# =============================================================================
def calibrate(cs_real: pd.DataFrame, obs_real: pd.DataFrame, ORGANISMS, true_mu) -> dict:
    cal = {}
    for org, prof in ORGANISMS.items():
        if org not in set(cs_real.organism):
            continue
        c = cs_real[cs_real.organism == org]
        o = obs_real[obs_real.organism == org]
        ok = c[c.usable_for_secondary_model]
        k = np.nan
        if len(ok) >= 8:
            tm = np.array(
                [
                    float(true_mu(prof, r.temperature_celsius, r.ph, r.water_activity))
                    for r in ok.itertuples()
                ]
            )
            ratio = ok["fit_mu_max_per_h"].to_numpy()[tm > 1e-3] / tm[tm > 1e-3]
            if len(ratio) >= 8:
                k = float(np.clip(np.median(ratio), 0.2, 3.0))
        # replicate scatter (SD of counts at identical times within a curve)
        sd = []
        for _, g in o.dropna(subset=["log10_count"]).groupby(["curve_id", "time_hours"]):
            if len(g) >= 2:
                sd.append(g["log10_count"].std(ddof=1))
        sd_med = float(np.clip(np.nanmedian(sd), 0.05, 0.6)) if len(sd) >= 20 else 0.15
        h0 = (ok["fit_lag_h"] * ok["fit_mu_max_per_h"]).median() if len(ok) else np.nan
        cal[org] = dict(
            k=k if np.isfinite(k) else 1.0,
            sd=sd_med,
            h0=float(h0) if np.isfinite(h0) else 2.0,
            n_cal=int(len(ok)),
            typ_ph=float(o["ph"].median()) if o["ph"].notna().any() else prof.phopt,
            typ_aw=(
                float(o["water_activity"].median()) if o["water_activity"].notna().any() else 0.99
            ),
        )
    return cal


def synth_curve(donor_obs: pd.DataFrame, donor_cs: pd.Series, prof, cal, true_mu, baranyi, rng):
    """Return (T, pH, aw, times, log10 values) for one synthetic curve modelled on `donor`."""
    T0, ph0, aw0 = donor_cs.temperature_celsius, donor_cs.ph, donor_cs.water_activity
    # Real conditions sit on lab-design grids (aw 0.997, pH 7.0 ...), so most synthetic curves keep the
    # donor's exact design and only ~35% get a small off-grid perturbation.
    T, ph, aw = T0, ph0, aw0
    if pd.notna(T0) and rng.random() < 0.35:
        T = round(float(T0 + rng.normal(0, 0.5)), 1)
    if pd.notna(ph0) and rng.random() < 0.35:
        ph = round(float(np.clip(ph0 + rng.normal(0, 0.05), 2.5, 10.5)), 2)
    if pd.notna(aw0) and rng.random() < 0.35:
        aw = round(float(np.clip(aw0 + rng.normal(0, 0.001), 0.5, 0.999)), 3)
    T = float(T) if pd.notna(T) else np.nan
    ph = float(ph) if pd.notna(ph) else np.nan
    aw = float(aw) if pd.notna(aw) else np.nan

    t = np.sort(donor_obs["time_hours"].to_numpy())
    scale = float(np.exp(rng.normal(0, 0.15)))
    t = np.round(np.where(t > 0, t * scale, 0.0), 2)
    tmax = max(float(t.max()), 1e-6)
    y0 = float(
        np.clip(
            (donor_cs.y_initial if pd.notna(donor_cs.y_initial) else 3.0) + rng.normal(0, 0.12),
            0.3,
            9.5,
        )
    )
    beh = donor_cs.behavior
    sd = cal["sd"] * float(np.exp(rng.normal(0, 0.3)))
    gen = dict(gen_behavior=beh, gen_mu_max_per_h=np.nan, gen_lag_h=np.nan)

    # effective (T, pH, aw) for the mechanistic prior when a value is unknown
    T_e = T if np.isfinite(T) else (T0 if pd.notna(T0) else 20.0)
    ph_e = ph if np.isfinite(ph) else cal["typ_ph"]
    aw_e = aw if np.isfinite(aw) else cal["typ_aw"]

    if beh == "growth" and tmax > 0:
        mu_prof = float(true_mu(prof, T_e, ph_e, aw_e)) * cal["k"]
        if bool(donor_cs.fit_ok) and np.isfinite(donor_cs.fit_mu_max_per_h):
            base = float(true_mu(prof, T0 if pd.notna(T0) else T_e, ph_e, aw_e)) * cal["k"]
            adj = np.clip(mu_prof / base, 0.5, 2.0) if base > 1e-6 else 1.0
            mu = donor_cs.fit_mu_max_per_h * adj * float(np.exp(rng.normal(0, 0.2)))
            h0 = max(0.2, donor_cs.fit_lag_h * donor_cs.fit_mu_max_per_h)
            incr = donor_cs.fit_increase_log10
        else:
            emp = LN10 * max(donor_cs.peak_rise_log10, 0.3) / tmax * 2.0
            mu = (mu_prof if mu_prof > 1e-3 else min(emp, 3.0)) * float(np.exp(rng.normal(0, 0.25)))
            h0 = max(0.3, rng.normal(cal["h0"], 0.5))
            incr = donor_cs.peak_rise_log10
        mu = float(np.clip(mu, 2e-3, 4.0))
        lam = float(min(h0 / mu * np.exp(rng.normal(0, 0.15)), 0.6 * tmax))
        incr = float(np.clip(incr * np.exp(rng.normal(0, 0.1)), 0.4, 10.5 - y0))
        y = y0 + baranyi(t, incr * LN10, mu, lam) / LN10
        gen.update(gen_mu_max_per_h=mu, gen_lag_h=lam)
    elif beh == "decline":
        total = float(max(donor_cs.max_fall_log10, 0.3) * np.exp(rng.normal(0, 0.2)))
        if (
            donor_cs.n_below_detection > 0
        ):  # donor hit the detection floor -> let the synthetic curve cross it too
            total = max(total, y0 + float(rng.uniform(0.05, 1.0)))
        p = float(rng.uniform(0.7, 1.4))
        y = y0 - total * (t / tmax) ** p
    else:  # flat / single_time
        drift = rng.normal(0, 0.05)
        y = y0 + drift * (t / tmax)
    y = y + rng.normal(0, sd, len(t))
    return T, ph, aw, t, np.round(np.clip(y, -5, 11.0), 2) + 0.0, gen


def generate_synthetic(obs_real, cs_real, need, ORGANISMS, true_mu, baranyi, rng, cal):
    out_rows, gen_params = [], {}
    counter = 0
    for org, n_need in need.items():
        if n_need <= 0:
            continue
        prof = ORGANISMS[org]
        pool = cs_real[(cs_real.organism == org) & (cs_real.n_rows >= 2)].reset_index(drop=True)
        by_curve = {cid: g for cid, g in obs_real[obs_real.organism == org].groupby("curve_id")}
        remaining, k = n_need, 0
        while remaining > 0:
            cand = pool if remaining > 12 else pool[pool.n_rows <= remaining]
            if cand.empty:
                cand = pool
            d = cand.iloc[int(rng.integers(len(cand)))]
            dobs = by_curve[d.curve_id]
            T, ph, aw, t, y, gen = synth_curve(dobs, d, prof, cal[org], true_mu, baranyi, rng)
            if len(t) > remaining:
                t, y = t[:remaining], y[:remaining]
            k += 1
            sid = f"SYN_{slug(org)}_{k:05d}"
            cid = f"{slug(org)}__{sid}"
            below = y < 0
            for i in range(len(t)):
                counter += 1
                out_rows.append(
                    dict(
                        observation_id=f"SYNOBS{counter:06d}",
                        curve_id=cid,
                        record_id=sid,
                        organism=org,
                        organism_source_label=org,
                        data_origin="synthetic_calibrated",
                        is_synthetic=True,
                        synthetic_donor_curve_id=d.curve_id,
                        food_category=d.food_category,
                        food_name=d.food_name,
                        temperature_celsius=T,
                        ph=ph,
                        water_activity=aw,
                        time_hours=float(t[i]),
                        log10_count_source_value=float(SENTINEL if below[i] else y[i]),
                        log10_count=np.nan if below[i] else float(y[i]),
                        below_detection_limit=bool(below[i]),
                        aw_implausible_removed=False,
                        nacl_pct=d.nacl_pct,
                        assumed_ph=d.assumed_ph,
                        assumed_aw=d.assumed_aw,
                        assumed_temperature=d.assumed_temperature,
                        assumed_initial_count=d.assumed_initial_count,
                        conditions_notes=pd.NA,
                        note_cut_minced=pd.NA,
                        note_raw=pd.NA,
                        note_shaken=pd.NA,
                        note_mixed_inoculation=pd.NA,
                        source_file="SYNTHETIC",
                        duplicate_export_occurrences=1,
                    )
                )
            remaining -= len(t)
            gen_params[cid] = gen
    return pd.DataFrame(out_rows), gen_params


def complete_row_splits(growth_rates: pd.DataFrame) -> pd.DataFrame:
    """Give real growth-rate rows that have no split one (rows from ComBase summary rates have no curve to
    inherit from). Deterministic 70/15/15 by (organism, record_id) so a record never straddles splits;
    rows that already have a split are never changed."""
    import hashlib

    g = growth_rates.copy()
    key = g["organism"].astype(str) + "||" + g["record_id"].astype(str)
    real = ~g["is_synthetic"].astype(bool)
    known = (
        g.loc[real & g["split"].notna()].assign(k=key).drop_duplicates("k").set_index("k")["split"]
    )

    def bucket(k):
        if k in known.index:
            return known[k]
        h = int(hashlib.sha256(k.encode()).hexdigest(), 16) % 10_000 / 10_000
        return "train" if h < 0.70 else ("validation" if h < 0.85 else "test")

    miss = real & g["split"].isna()
    hashes = key.map(lambda k: int(hashlib.sha256(k.encode()).hexdigest(), 16) % 10_000)
    g.loc[miss, "split"] = [bucket(k) for k in key[miss]]
    # Guard: an organism with enough records must still have real test rows. If hashing left none,
    # promote the newly assigned records with the highest hash to "test" (~15% of its records).
    for org, d in g[real].groupby("organism"):
        n_records = key[d.index].nunique()
        if n_records >= 10 and not (d["split"] == "test").any():
            cand = d.index[miss.loc[d.index]]
            order = hashes.loc[cand].sort_values(ascending=False).index
            chosen = key[order].drop_duplicates().iloc[: max(1, round(0.15 * n_records))]
            g.loc[cand[key[cand].isin(chosen)], "split"] = "test"
    return g


# =============================================================================
# 4. VALIDATION
# =============================================================================
def validate(obs_all: pd.DataFrame, cs_all: pd.DataFrame, cs_syn_gen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for org, g in obs_all.groupby("organism"):
        r, s = g[~g.is_synthetic], g[g.is_synthetic]
        row = dict(organism=org, n_real=len(r), n_synthetic=len(s))
        for col, label in [
            ("temperature_celsius", "T"),
            ("ph", "pH"),
            ("water_activity", "aw"),
            ("time_hours", "time"),
            ("log10_count", "count"),
        ]:
            a, b = r[col].dropna(), s[col].dropna()
            if len(a) > 20 and len(b) > 20:
                ks = stats.ks_2samp(a, b)
                row[f"KS_{label}"] = round(float(ks.statistic), 3)
        cr, cs_ = (
            cs_all[(cs_all.organism == org) & (~cs_all.is_synthetic)],
            cs_all[(cs_all.organism == org) & cs_all.is_synthetic],
        )
        for b in ["growth", "decline", "flat", "single_time"]:
            row[f"real_{b}_curve_share"] = (
                round(float((cr.behavior == b).mean()), 3) if len(cr) else np.nan
            )
            row[f"syn_{b}_curve_share"] = (
                round(float((cs_.behavior == b).mean()), 3) if len(cs_) else np.nan
            )
        sg = cs_syn_gen[
            (cs_syn_gen.organism == org) & cs_syn_gen.fit_ok & cs_syn_gen.gen_mu_max_per_h.notna()
        ]
        if len(sg) >= 5:
            ratio = sg.fit_mu_max_per_h / sg.gen_mu_max_per_h
            row["syn_mu_recovery_median_ratio"] = round(float(ratio.median()), 3)
            row["syn_mu_recovery_n"] = int(len(sg))
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--rates-xlsx", required=True)
    ap.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, a.project_root)
    from src.datasets import ORGANISMS, true_mu
    from src.models.primary import FitError, baranyi, fit_primary_curve

    rng = np.random.default_rng(SEED)
    log: list = []

    raw = pd.read_csv(a.raw)
    real = clean_real(raw, log)
    real = add_bands(real)
    cs_real = fit_curves(real, (fit_primary_curve, FitError))
    cs_real["split"] = assign_split(cs_real, rng)
    real = attach_curve_columns(real, cs_real)
    log.append(("curves (organism+record) found", len(cs_real), ""))
    log.append(
        (
            "growth curves with accepted primary fit",
            int(cs_real.fit_ok.sum()),
            "Gompertz, R2>=0.8, rel SE<=50%",
        )
    )
    log.append(
        (
            "curves usable for secondary model (fit ok + T/pH/aw)",
            int(cs_real.usable_for_secondary_model.sum()),
            "",
        )
    )

    # ---- synthetic top-up ---------------------------------------------------------
    cal = calibrate(cs_real, real, ORGANISMS, true_mu)
    have = real.groupby("organism").size()
    need = {o: TARGET_PER_ORGANISM - int(have.get(o, 0)) for o in have.index}
    syn, gen_params = generate_synthetic(real, cs_real, need, ORGANISMS, true_mu, baranyi, rng, cal)
    syn = add_bands(syn)
    cs_syn = fit_curves(syn, (fit_primary_curve, FitError))
    # generating parameters for validation
    gp = pd.DataFrame(gen_params).T
    cs_syn["gen_behavior"] = cs_syn.curve_id.map(gp["gen_behavior"])
    cs_syn["gen_mu_max_per_h"] = pd.to_numeric(
        cs_syn.curve_id.map(gp["gen_mu_max_per_h"]), errors="coerce"
    )
    cs_syn["gen_lag_h"] = pd.to_numeric(cs_syn.curve_id.map(gp["gen_lag_h"]), errors="coerce")
    cs_syn["split"] = cs_syn["synthetic_donor_curve_id"].map(cs_real.set_index("curve_id")["split"])
    syn = attach_curve_columns(syn, cs_syn)
    log.append(("synthetic rows generated", len(syn), "to bring each organism to exactly 4,000"))

    # ---- combine -------------------------------------------------------------------
    cols = [
        "observation_id",
        "curve_id",
        "record_id",
        "organism",
        "organism_source_label",
        "data_origin",
        "is_synthetic",
        "synthetic_donor_curve_id",
        "food_category",
        "food_name",
        "temperature_celsius",
        "temperature_band",
        "ph",
        "ph_class",
        "water_activity",
        "aw_class",
        "time_hours",
        "log10_count",
        "log10_count_source_value",
        "below_detection_limit",
        "log10_change_from_initial",
        "ln_change_from_initial",
        "nacl_pct",
        "assumed_ph",
        "assumed_aw",
        "assumed_temperature",
        "assumed_initial_count",
        "metadata_complete",
        "curve_behavior",
        "curve_n_points",
        "curve_duration_h",
        "curve_log10_initial",
        "curve_mu_max_per_h",
        "curve_lag_h",
        "curve_fit_r2",
        "curve_growth_fit_ok",
        "curve_slope_log10_per_h",
        "usable_for_primary_fit",
        "usable_for_secondary_model",
        "include_for_modeling",
        "split",
        "qc_flags",
        "aw_implausible_removed",
        "note_cut_minced",
        "note_raw",
        "note_shaken",
        "note_mixed_inoculation",
        "conditions_notes",
        "source_file",
        "duplicate_export_occurrences",
    ]
    real_out, syn_out = real[cols].copy(), syn[cols].copy()
    combined = pd.concat([real_out, syn_out], ignore_index=True)
    cs_all = pd.concat([cs_real, cs_syn], ignore_index=True)

    # ---- model-ready growth-rate table (project DATASET_COLUMNS + extras) ----------
    rates = pd.read_excel(a.rates_xlsx, sheet_name="Numeric_Rates_Review")
    rates = rates.rename(
        columns={
            "Organism": "organism_label",
            "Record ID": "record_id",
            "Temperature (C)": "T",
            "Aw": "aw",
            "pH": "ph_",
            "source_rate_numeric_log10_per_h": "r10",
        }
    )
    rates["organism"] = rates["organism_label"].map(ORG_MAP)
    for c in ["T", "aw", "ph_", "r10"]:
        rates[c] = pd.to_numeric(rates[c], errors="coerce")
    rates = rates.dropna(subset=["organism", "T", "aw", "ph_", "r10"])
    rates = rates[rates.r10 > 0]
    summ = pd.DataFrame(
        dict(
            organism=rates.organism,
            temperature_celsius=rates["T"],
            ph=rates.ph_,
            water_activity=rates.aw,
            mu_max_per_h=rates.r10 * LN10,
            lag_h=np.nan,
            record_id=rates.record_id,
            food_category=rates["Food category"],
            assumed=rates["Assumed"],
            source="combase_summary_rate",
            is_synthetic=False,
            synthetic_donor_curve_id=pd.NA,
            split=pd.NA,
        )
    )
    sp = cs_real.set_index(["organism", "record_id"])["split"]
    summ["split"] = [sp.get((o, r), pd.NA) for o, r in zip(summ.organism, summ.record_id)]
    have_summary = set(zip(summ.organism, summ.record_id))

    def fitted(cs, source):
        f = cs[cs.usable_for_secondary_model].copy()
        f = (
            f[[(o, r) not in have_summary for o, r in zip(f.organism, f.record_id)]]
            if not source.startswith("synthetic")
            else f
        )
        return pd.DataFrame(
            dict(
                organism=f.organism,
                temperature_celsius=f.temperature_celsius,
                ph=f.ph,
                water_activity=f.water_activity,
                mu_max_per_h=f.fit_mu_max_per_h,
                lag_h=f.fit_lag_h,
                record_id=f.record_id,
                food_category=f.food_category,
                assumed=np.where(
                    f.assumed_ph.fillna(False) | f.assumed_aw.fillna(False), "assumed", ""
                ),
                source=source,
                is_synthetic=f.is_synthetic,
                synthetic_donor_curve_id=f.synthetic_donor_curve_id,
                split=f.split,
            )
        )

    growth_rates = pd.concat(
        [
            summ,
            fitted(cs_real, "fitted_from_real_curve"),
            fitted(cs_syn, "synthetic_calibrated_curve_fit"),
        ],
        ignore_index=True,
    )
    growth_rates = growth_rates[growth_rates.mu_max_per_h.between(1e-3, 6.0)].reset_index(drop=True)
    growth_rates = complete_row_splits(growth_rates)

    # ---- coverage -------------------------------------------------------------------
    cov = (
        combined.groupby(["organism"])
        .agg(
            total_observations=("observation_id", "size"),
            real_observations=("is_synthetic", lambda s: int((~s).sum())),
            synthetic_observations=("is_synthetic", "sum"),
            curves=("curve_id", "nunique"),
        )
        .reset_index()
    )
    cov["target"] = TARGET_PER_ORGANISM
    cov["target_met"] = cov.total_observations == cov.target
    cov["synthetic_share_pct"] = (100 * cov.synthetic_observations / cov.total_observations).round(
        1
    )
    g = (
        cs_all[cs_all.usable_for_secondary_model]
        .groupby(["organism", "is_synthetic"])
        .size()
        .unstack(fill_value=0)
    )
    g.columns = [
        "usable_growth_curves_real" if c is False else "usable_growth_curves_synthetic"
        for c in g.columns
    ]
    cov = cov.merge(g.reset_index(), on="organism", how="left").fillna(
        {"usable_growth_curves_real": 0, "usable_growth_curves_synthetic": 0}
    )

    # validation (needs generating params -> recover from fits only; recovery measured against donor-based mu)
    val = validate(combined, cs_all, cs_syn)
    cal_df = pd.DataFrame(cal).T.reset_index().rename(columns={"index": "organism"})

    # ---- write ---------------------------------------------------------------------
    real_out.to_csv(out / "microbesim_clean_real_observations.csv", index=False)
    syn_out.to_csv(out / "microbesim_synthetic_observations.csv", index=False)
    combined.to_csv(out / "microbesim_combined_observations_40k.csv", index=False)
    cs_all.to_csv(out / "microbesim_curve_summary.csv", index=False)
    growth_rates.to_csv(out / "microbesim_growth_rates_model_ready.csv", index=False)
    cov.to_csv(out / "microbesim_organism_coverage_updated.csv", index=False)
    val.to_csv(out / "microbesim_synthetic_validation.csv", index=False)
    cal_df.to_csv(out / "microbesim_synthetic_calibration.csv", index=False)
    pd.DataFrame(log, columns=["step", "rows_affected", "detail"]).to_csv(
        out / "microbesim_cleaning_log.csv", index=False
    )
    print(cov.to_string())
    print(val.to_string())
    print(cal_df.to_string())
    print(pd.DataFrame(log, columns=["step", "n", "detail"]).to_string())
    print(
        "growth-rate rows:",
        growth_rates.groupby(["organism", "is_synthetic"]).size().unstack(fill_value=0),
    )


if __name__ == "__main__":
    main()
