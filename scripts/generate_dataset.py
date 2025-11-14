"""Utility for generating a synthetic SNP dataset.

The script mimics a genotype matrix (X) and a quantitative phenotype (y)
by sampling values from simple distributions.  The defaults replicate the
behaviour of the original ``archivo.py`` script provided by the user, but the
implementation has been expanded to expose command line switches, add
reproducibility, and provide better diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


def _parse_effects(raw_effects: str | None, n_snps: int) -> Mapping[int, float]:
    """Parse the ``--effects`` argument into a dictionary.

    The value is expected to be a JSON object where each key is the index of an
    SNP (0-indexed) and each value is the effect size.  When the argument is not
    provided a trio of sensible defaults is returned.
    """

    if raw_effects is None:
        return {3: 1.5, 10: -2.0, 25: 3.0}

    try:
        parsed = json.loads(raw_effects)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(
            "--effects must be a valid JSON object, e.g. '{\"0\": 1.2}'"
        ) from exc

    effects: dict[int, float] = {}
    for key, value in parsed.items():
        try:
            idx = int(key)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise argparse.ArgumentTypeError(
                "SNP indices must be integers in the JSON payload"
            ) from exc
        if idx < 0 or idx >= n_snps:
            raise argparse.ArgumentTypeError(
                f"SNP index {idx} out of bounds for {n_snps} features"
            )
        effects[idx] = float(value)
    return effects


def generate_dataset(
    *,
    n_individuals: int,
    n_snps: int,
    effects: Mapping[int, float],
    noise_std: float,
    seed: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Generate a synthetic genotype/phenotype dataset.

    Parameters
    ----------
    n_individuals:
        Number of samples (rows) in the resulting dataset.
    n_snps:
        Number of SNP features.
    effects:
        Mapping of ``{column_index: effect_size}`` describing which SNPs truly
        influence the phenotype.
    noise_std:
        Standard deviation of the Gaussian noise added to the phenotype.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    X_df, y_df, beta
        The genotype matrix, phenotype series and the underlying coefficients
        used for generation.
    """

    rng = np.random.default_rng(seed)

    X = rng.integers(0, 3, size=(n_individuals, n_snps))
    beta = np.zeros(n_snps)
    for idx, value in effects.items():
        beta[idx] = value

    noise = rng.normal(0, noise_std, size=n_individuals)
    y = X @ beta + noise

    feature_names = [f"snp_{i:03d}" for i in range(n_snps)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_df = pd.DataFrame({"y": y})
    return X_df, y_df, beta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Directory where CSV files will be stored")
    parser.add_argument("--n-individuals", type=int, default=100, help="Number of samples (default: 100)")
    parser.add_argument("--n-snps", type=int, default=50, help="Number of SNP features (default: 50)")
    parser.add_argument(
        "--effects",
        type=str,
        default=None,
        help="JSON object with SNP indices as keys and effect sizes as values",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=2.0,
        help="Standard deviation of the Gaussian noise added to the phenotype",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed for the pseudo random number generator (default: 12345)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    effects = _parse_effects(args.effects, args.n_snps)
    X_df, y_df, beta = generate_dataset(
        n_individuals=args.n_individuals,
        n_snps=args.n_snps,
        effects=effects,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    x_path = args.output / "X.csv"
    y_path = args.output / "y.csv"
    X_df.to_csv(x_path, index=False)
    y_df.to_csv(y_path, index=False)

    print(f"Dataset written to {x_path} and {y_path}")
    print("True beta coefficients (non-zero entries):")
    for idx, value in effects.items():
        print(f"  snp_{idx:03d}: {value:+.3f}")


if __name__ == "__main__":
    main()
