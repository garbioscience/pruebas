"""Train and evaluate a Ridge regression model on SNP data.

This script is an enhanced version of the provided ``1.py`` example.  It adds
basic validation, allows the dataset and output directory to be configured, and
reports quantitative metrics in addition to the diagnostic plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_dataset(x_path: Path, y_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the genotype matrix and phenotype vector from CSV files."""

    X_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)
    if "y" not in y_df.columns:
        raise ValueError("The y.csv file must contain a 'y' column")
    return X_df.values, y_df["y"].values


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, output: Path) -> Path:
    """Create a scatter plot comparing real and predicted phenotypes."""

    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.75)
    ax.set_xlabel("Fenotipo real (y)")
    ax.set_ylabel("Fenotipo predicho")
    ax.set_title("Predicción vs Real")
    lims = np.array([*ax.get_xlim(), *ax.get_ylim()])
    min_lim, max_lim = lims.min(), lims.max()
    ax.plot([min_lim, max_lim], [min_lim, max_lim], "k--", linewidth=1)
    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)

    path = output / "grafico_prediccion.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_coefficients(coefficients: np.ndarray, output: Path) -> Path:
    """Bar plot of the Ridge coefficients (importance of SNPs)."""

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(coefficients)), coefficients)
    ax.set_xlabel("Índice de SNP")
    ax.set_ylabel("Coeficiente Beta")
    ax.set_title("Importancia de SNPs")

    path = output / "importancia_snps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Directorio que contiene X.csv e y.csv")
    parser.add_argument("--alpha", type=float, default=1.0, help="Parámetro de regularización del modelo Ridge")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Proporción del conjunto de test (default: 0.2)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out"),
        help="Directorio donde se guardarán los resultados (default: ./out)",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Semilla para la división train/test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x_path = args.data / "X.csv"
    y_path = args.data / "y.csv"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError("El directorio de datos debe contener X.csv e y.csv")

    X, y = load_dataset(x_path, y_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    model = Ridge(alpha=args.alpha)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "r2_train": r2_score(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred_test),
        "rmse_train": mean_squared_error(y_train, y_pred_train, squared=False),
        "rmse_test": mean_squared_error(y_test, y_pred_test, squared=False),
    }

    print("Dimensión de X:", X.shape)
    print("Ejemplo de fila de X:", X[0])
    print("\nCoeficientes beta (importancia de SNPs):")
    print(model.coef_)
    print("\nMétricas del modelo:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    args.output.mkdir(parents=True, exist_ok=True)
    scatter_path = plot_predictions(y_test, y_pred_test, args.output)
    coef_path = plot_coefficients(model.coef_, args.output)
    print("\nGráficos generados:")
    print(f"  {scatter_path}")
    print(f"  {coef_path}")


if __name__ == "__main__":
    main()
