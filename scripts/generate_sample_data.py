"""Regenerate the sample files in ./data (deterministic)."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets import generate_growth_dataset
from src.models.primary import gompertz
from src.simulation.scenarios import cold_chain_break

DATA = Path(__file__).resolve().parents[1] / "data"


def kinetic_samples() -> pd.DataFrame:
    """Synthetic growth curves for the organisms included in the sample dataset."""
    rng = np.random.default_rng(7)
    t = np.round(np.arange(0.0, 24.01, 0.5), 2)
    rows = []
    organism_configs = [
        ("Escherichia coli", "EC", 37.0, 30.0, 25.0, 0.95, 0.58, 0.38),
        ("Salmonella spp.", "SA", 37.0, 30.0, 25.0, 0.95, 0.55, 0.30),
        ("Listeria monocytogenes", "LM", 30.0, 20.0, 10.0, 0.48, 0.30, 0.12),
        ("Staphylococcus aureus", "ST", 37.0, 30.0, 20.0, 0.75, 0.45, 0.20),
        ("Pseudomonas spp.", "PS", 25.0, 15.0, 5.0, 0.70, 0.38, 0.10),
        ("Bacillus cereus", "BC", 35.0, 25.0, 15.0, 0.82, 0.45, 0.18),
        ("Campylobacter jejuni", "CJ", 42.0, 37.0, 32.0, 0.42, 0.28, 0.12),
        ("Clostridium perfringens", "CP", 43.0, 37.0, 25.0, 1.00, 0.60, 0.22),
        ("Clostridium botulinum non-proteolytic", "CBN", 25.0, 15.0, 8.0, 0.38, 0.18, 0.07),
        ("Clostridium botulinum proteolytic", "CBP", 35.0, 25.0, 15.0, 0.52, 0.28, 0.10),
        ("Yersinia enterocolitica", "YE", 28.0, 20.0, 10.0, 0.45, 0.24, 0.10),
    ]
    for organism, code, temp1, temp2, temp3, mu1, mu2, mu3 in organism_configs:
        for index, (temp, mu) in enumerate(((temp1, mu1), (temp2, mu2), (temp3, mu3)), 1):
            sid = f"{code}{index}"
            y = gompertz(t, 6.2 - index * 0.2, mu, 1.5 + index * 1.2) + rng.normal(0, 0.06, len(t))
            log10 = 3.0 + y / np.log(10.0)
            rows.append(
                pd.DataFrame(
                    {
                        "series_id": sid,
                        "organism": organism,
                        "temperature_celsius": temp,
                        "ph": 7.0,
                        "water_activity": 0.99,
                        "time_hours": t,
                        "measurement": np.round(log10, 4),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def cold_chain_sample() -> pd.DataFrame:
    pts = cold_chain_break(
        fridge_c=4.0, excursion_c=28.0, excursion_start=30.0, excursion_hours=10.0, hours=120.0
    )
    return pd.DataFrame(
        {
            "timestamp_hours": [p.timestamp_hours for p in pts],
            "temperature_celsius": [round(p.temperature_celsius, 2) for p in pts],
        }
    )


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    kinetic_samples().to_csv(DATA / "kinetic_curve_sample.csv", index=False)
    cold_chain_sample().to_csv(DATA / "cold_chain_sample.csv", index=False)
    from src.datasets import ORGANISMS

    for index, organism in enumerate(ORGANISMS, start=1):
        filename = re.sub(r"[^a-z0-9]+", "_", organism.lower()).strip("_") + ".csv"
        generate_growth_dataset(organism, n=700, seed=40 + index).to_csv(
            DATA / "synthetic" / filename, index=False
        )
    print("wrote sample files to", DATA)
