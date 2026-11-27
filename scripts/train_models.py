"""Train and benchmark growth-rate models on bundled datasets."""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import (
    load_expanded_growth_dataset,
    load_real_growth_dataset,
    load_synthetic_growth_dataset,
    real_organisms,
    synthetic_organisms,
)
from src.model_cache import load_or_train
from src.models.benchmark import available_model_keys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            'Examples: --source both; --source real --organism "Escherichia coli"; '
            "--source synthetic --models ratkowsky rf"
        ),
    )
    parser.add_argument(
        "--source",
        choices=("real", "synthetic", "both", "expanded"),
        default="both",
        help=(
            "Dataset source(s) to train on (default: both). "
            "'expanded' uses data/expanded with its per-curve train/validation/test splits."
        ),
    )
    parser.add_argument(
        "--organism",
        action="append",
        help="Organism to train; repeat to select multiple. Defaults to all eligible organisms.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=available_model_keys(),
        default=available_model_keys(),
        help="Model keys to benchmark (default: all available models).",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Fit models again even when a compatible cached benchmark exists.",
    )
    return parser


def _expanded_organisms() -> List[str]:
    frame = load_expanded_growth_dataset(include_synthetic=False)
    counts = frame.groupby("organism").size()
    return sorted(counts[counts >= 20].index)


def _selected_datasets(source: str, organisms: List[str] | None) -> List[Tuple[str, str]]:
    if source == "expanded":
        available = _expanded_organisms()
        requested = available if organisms is None else organisms
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"expanded data has no eligible dataset for: {', '.join(missing)}")
        return [("expanded", organism) for organism in requested]
    loaders = {
        "real": (real_organisms(), load_real_growth_dataset),
        "synthetic": (synthetic_organisms(), load_synthetic_growth_dataset),
    }
    sources = ("real", "synthetic") if source == "both" else (source,)
    selected = []
    for name in sources:
        available, _ = loaders[name]
        requested = available if organisms is None else organisms
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"{name} data has no eligible dataset for: {', '.join(missing)}")
        selected.extend((name, organism) for organism in requested)
    if not selected:
        raise ValueError(f"No eligible datasets found for source '{source}'.")
    return selected


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        datasets = _selected_datasets(args.source, args.organism)
    except ValueError as exc:
        parser.error(str(exc))

    failed = False
    for source, organism in datasets:
        if source == "expanded":
            # Train on real rows only and on real+synthetic; both are scored on REAL test rows.
            frame = load_expanded_growth_dataset(organism)
            for training_source in ("real", "real+synthetic"):
                print(f"\n[expanded/{training_source}] {organism}: {len(frame)} rows")
                result = load_or_train(
                    frame,
                    args.models,
                    force_retrain=args.force_retrain,
                    training_source=training_source,
                )
                action = (
                    "Loaded cached benchmark" if result.loaded_from_disk else "Trained and saved"
                )
                print(f"{action} ({result.validation_description}).")
                if result.table.empty:
                    failed = True
                    print("No selected model could be trained.")
                else:
                    columns = [c for c in ("Model", "R2", "RMSE", "Af", "Bf") if c in result.table]
                    print(
                        result.table[columns].to_string(
                            index=False, float_format=lambda v: f"{v:.4g}"
                        )
                    )
                for key, message in result.errors.items():
                    print(f"Model error [{key}]: {message}")
            continue
        frame = (
            load_real_growth_dataset(organism)
            if source == "real"
            else load_synthetic_growth_dataset(organism)
        )
        groups = None
        group_name = "supplied labels"
        if "food_category" in frame:
            groups = frame["food_category"].fillna("Unknown").astype(str).to_numpy()
            group_name = "food category"

        print(f"\n[{source}] {organism}: {len(frame)} rows")
        result = load_or_train(
            frame,
            args.models,
            groups=groups,
            group_name=group_name,
            force_retrain=args.force_retrain,
        )
        action = "Loaded cached benchmark" if result.loaded_from_disk else "Trained and saved"
        print(f"{action} ({result.validation_description}).")
        if result.table.empty:
            failed = True
            print("No selected model could be trained.")
        else:
            columns = [
                column for column in ("Model", "R2", "RMSE", "Af", "Bf") if column in result.table
            ]
            print(
                result.table[columns].to_string(
                    index=False, float_format=lambda value: f"{value:.4g}"
                )
            )
        for key, message in result.errors.items():
            print(f"Model error [{key}]: {message}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
