"""
experiments.py
==============
Command-line entry-point that replicates every experiment from
``notebooks/01_neural_network_from_scratch.ipynb`` and
``notebooks/02_transferLearning_vs_scratch.ipynb`` as reproducible,
importable Python functions.

Usage
-----
Run all experiments back-to-back::

    python src/experiments.py all \\
        --train-dir data/train \\
        --test-dir  data/test  \\
        --output-dir results

Run only the hidden-unit sweep (Part B)::

    python src/experiments.py hidden-units \\
        --train-dir data/train \\
        --test-dir  data/test

Run only the depth + activation comparison (Parts C & D) and the
sklearn MLPClassifier baseline (Part E)::

    python src/experiments.py depth       --train-dir ... --test-dir ...
    python src/experiments.py activation  --train-dir ... --test-dir ...
    python src/experiments.py mlp         --train-dir ... --test-dir ...

Notebook 02 — transfer learning (needs digits train/test dirs)::

    python src/experiments.py transfer \\
        --train-dir data/train_digits \\
        --test-dir  data/test_digits  \\
        --pretrained-model model/pretrained_consonant_model.pkl

Common optional flags (all subcommands)
----------------------------------------
--epochs        Max training epochs             (default: 50)
--batch-size    Mini-batch size                 (default: 32)
--lr            Learning rate                   (default: 0.01)
--tol           Early-stopping tolerance        (default: 1e-4)
--seed          Global random seed              (default: 42)
--save-model    Path to pickle the best model   (default: off)
--output-dir    Directory for CSV + PNG results (default: results/)
--no-plots      Skip matplotlib figure output   (flag)
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier

# Support both  `python src/experiments.py …`  and  `python -m src.experiments`
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_utils import load_devanagari_data, shuffle_data
from generic_neural_network import (
    GenericNeuralNetwork,
    relu,
    relu_deriv,
    sigmoid,
    sigmoid_deriv,
)


@dataclass
class ExperimentConfig:

    epochs: int = 50
    batch_size: int = 32
    lr: float = 0.01
    tol: float = 1e-4
    seed: int = 42
    output_dir: str = "results"
    save_model: str | None = None
    no_plots: bool = False



def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_csv(
    path: str,
    headers: list[str],
    rows: list[list],
) -> None:
    #Write a minimal CSV without requiring pandas.
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(headers) + "\n")
        for row in rows:
            fh.write(",".join(str(v) for v in row) + "\n")
    print(f"  [saved] {path}")


def _evaluate(
    model: GenericNeuralNetwork,
    X_train: NDArray[np.floating],
    Y_train: NDArray[np.floating],
    X_test: NDArray[np.floating],
    Y_test: NDArray[np.floating],
) -> tuple[float, float]:
    
    true_train = np.argmax(Y_train, axis=0)
    true_test = np.argmax(Y_test, axis=0)
    train_f1 = float(f1_score(true_train, model.predict(X_train), average="macro", zero_division=0))
    test_f1 = float(f1_score(true_test, model.predict(X_test), average="macro", zero_division=0))
    return train_f1, test_f1


def _maybe_show(cfg: ExperimentConfig, fig_path: str) -> None:
   #Save the current matplotlib figure and optionally display it.

    import matplotlib.pyplot as plt

    dest = os.path.join(cfg.output_dir, fig_path)
    plt.savefig(dest, bbox_inches="tight", dpi=150)
    print(f"  [saved] {dest}")
    if not cfg.no_plots:
        plt.show()
    plt.close()


# Experiment A – single sanity-check model (Part a of notebook)

def run_sanity_check(
    X_train: NDArray[np.floating],
    Y_train: NDArray[np.floating],
    X_test: NDArray[np.floating],
    Y_test: NDArray[np.floating],
    cfg: ExperimentConfig,
) -> dict[str, float]:
    """Train a single baseline model with one hidden layer of 100 units.

    This mirrors the initial single-model cell in the notebook (Part a),
    confirming that training and evaluation machinery is working before
    running the full sweeps.

    Returns
    dict[str, float]
        ``{"train_f1": ..., "test_f1": ...}``
    """
    print("Experiment: Sanity-check single model [100]")
    n, r = X_train.shape[0], Y_train.shape[0]
    model = GenericNeuralNetwork(n=n, hidden_layers=[100], r=r, seed=cfg.seed)
    model.train(X_train, Y_train, cfg.batch_size, cfg.epochs, cfg.lr, cfg.tol)

    train_f1, test_f1 = _evaluate(model, X_train, Y_train, X_test, Y_test)
    print(f"\nSanity check => Train F1: {train_f1:.4f} | Test F1: {test_f1:.4f}")
    return {"train_f1": train_f1, "test_f1": test_f1}


# Experiment B – varying hidden units (single hidden layer)

def run_hidden_units(
    X_train: NDArray[np.floating],
    Y_train: NDArray[np.floating],
    X_test: NDArray[np.floating],
    Y_test: NDArray[np.floating],
    cfg: ExperimentConfig,
    hidden_units_options: list[int] | None = None,
) -> list[dict]:
    """Sweep over single-hidden-layer networks of varying width (Part B).

    Trains one model per entry in *hidden_units_options*, evaluates macro-F1
    on both splits, and optionally saves a PNG line-plot and a CSV summary.

    hidden_units_options:
        List of hidden-unit counts to sweep.  Defaults to
        ``[1, 5, 10, 50, 100]`` 

    Returns:
        list[dict]
        One dict per configuration: ``{"hidden_units", "train_f1", "test_f1"}``.
    """
    if hidden_units_options is None:
        hidden_units_options = [1, 5, 10, 50, 100]

    print("Experiment B: Varying hidden units (single hidden layer, sigmoid)")

    n, r = X_train.shape[0], Y_train.shape[0]
    results: list[dict] = []

    for h in hidden_units_options:
        print(f"\nTraining Model with {h} hidden unit(s)")
        model = GenericNeuralNetwork(n=n, hidden_layers=[h], r=r, seed=cfg.seed)
        model.train(X_train, Y_train, cfg.batch_size, cfg.epochs, cfg.lr, cfg.tol)

        true_test = np.argmax(Y_test, axis=0)
        precision, recall, f1_per_class, _ = precision_recall_fscore_support(
            true_test, model.predict(X_test), zero_division=0
        )
        print("  Class-wise Test Metrics (first 5 classes):")
        for c in range(min(5, r)):
            print(
                f"    Class {c:2d} -> Precision: {precision[c]:.4f}  "
                f"Recall: {recall[c]:.4f}  F1: {f1_per_class[c]:.4f}"
            )

        train_f1, test_f1 = _evaluate(model, X_train, Y_train, X_test, Y_test)
        print(f"  -> Train Avg F1: {train_f1:.4f} | Test Avg F1: {test_f1:.4f}")
        results.append({"hidden_units": h, "train_f1": train_f1, "test_f1": test_f1})

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        x_vals = [r["hidden_units"] for r in results]
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=x_vals, y=[r["test_f1"] for r in results], marker="o", label="Test F1")
        sns.lineplot(x=x_vals, y=[r["train_f1"] for r in results], marker="x", label="Train F1")
        plt.title("Part B: Average F1 Score vs. Number of Hidden Units")
        plt.xlabel("Number of Hidden Units")
        plt.ylabel("Average F1 Score (Macro)")
        plt.xscale("log")
        plt.xticks(x_vals, labels=[str(v) for v in x_vals])
        plt.grid(True, which="both", ls="--", c="0.7")
        plt.legend()
        plt.tight_layout()
        _maybe_show(cfg, "part_b_hidden_units.png")
    except ImportError:
        print("  [skip] matplotlib / seaborn not installed; skipping plot.")

    # ---- CSV ----
    _save_csv(
        os.path.join(cfg.output_dir, "part_b_hidden_units.csv"),
        ["hidden_units", "train_f1", "test_f1"],
        [[r["hidden_units"], f"{r['train_f1']:.6f}", f"{r['test_f1']:.6f}"] for r in results],
    )

    return results


# Experiment C – varying depth (sigmoid)

def run_depth_sigmoid(
    X_train: NDArray[np.floating],
    Y_train: NDArray[np.floating],
    X_test: NDArray[np.floating],
    Y_test: NDArray[np.floating],
    cfg: ExperimentConfig,
    depth_configs: list[list[int]] | None = None,
) -> list[dict]:
    """Sweep over multi-layer sigmoid networks of varying depth (Part C).

    Parameters
    depth_configs:
        List of hidden-layer configurations to sweep. 
        ``[[512], [512,256], [512,256,128], [512,256,128,64]]``.

    Returns
    list[dict]
        One dict per configuration:
        ``{"hidden_layers", "depth", "train_f1", "test_f1"}``.
    """
    if depth_configs is None:
        depth_configs = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]

    print("\n" + "=" * 60)
    print("Experiment C: Varying depth — sigmoid activation")
    print("=" * 60)

    n, r = X_train.shape[0], Y_train.shape[0]
    results: list[dict] = []

    for h in depth_configs:
        depth = len(h)
        print(f"\nTraining Model with Depth {depth}: {h}")
        model = GenericNeuralNetwork(n=n, hidden_layers=h, r=r, seed=cfg.seed)
        model.train(X_train, Y_train, cfg.batch_size, cfg.epochs, cfg.lr, cfg.tol)

        true_test = np.argmax(Y_test, axis=0)
        precision, recall, f1_per_class, _ = precision_recall_fscore_support(
            true_test, model.predict(X_test), zero_division=0
        )
        print("  Class-wise Test Metrics (first 5 classes):")
        for c in range(min(5, r)):
            print(
                f"    Class {c:2d} -> Precision: {precision[c]:.4f}  "
                f"Recall: {recall[c]:.4f}  F1: {f1_per_class[c]:.4f}"
            )

        train_f1, test_f1 = _evaluate(model, X_train, Y_train, X_test, Y_test)
        print(f"  -> Train Avg F1: {train_f1:.4f} | Test Avg F1: {test_f1:.4f}")
        results.append({"hidden_layers": h, "depth": depth, "train_f1": train_f1, "test_f1": test_f1})

    # ---- Plot ----
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        depths = [r["depth"] for r in results]
        labels = [str(r["hidden_layers"]) for r in results]
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=depths, y=[r["test_f1"] for r in results], marker="o", label="Test F1")
        sns.lineplot(x=depths, y=[r["train_f1"] for r in results], marker="x", label="Train F1")
        plt.title("Part C: Average F1 Score vs. Network Depth (Sigmoid)")
        plt.xlabel("Depth (Number of Hidden Layers)")
        plt.ylabel("Average F1 Score (Macro)")
        plt.xticks(depths, labels=labels, rotation=45, ha="right")
        plt.grid(True, which="both", ls="--", c="0.7")
        plt.legend()
        plt.tight_layout()
        _maybe_show(cfg, "part_c_depth_sigmoid.png")
    except ImportError:
        print("  [skip] matplotlib / seaborn not installed; skipping plot.")

    _save_csv(
        os.path.join(cfg.output_dir, "part_c_depth_sigmoid.csv"),
        ["hidden_layers", "depth", "train_f1", "test_f1"],
        [
            [str(r["hidden_layers"]), r["depth"], f"{r['train_f1']:.6f}", f"{r['test_f1']:.6f}"]
            for r in results
        ],
    )

    return results

# Experiment D – varying depth (ReLU)

def run_depth_relu(
    X_train: NDArray[np.floating],
    Y_train: NDArray[np.floating],
    X_test: NDArray[np.floating],
    Y_test: NDArray[np.floating],
    cfg: ExperimentConfig,
    depth_configs: list[list[int]] | None = None,
) -> list[dict]:
    """Sweep over multi-layer ReLU networks of varying depth (Part D).

    Identical structure to :func:`run_depth_sigmoid` but uses ReLU hidden
    activations and He weight initialisation (handled transparently by
    :class:`GenericNeuralNetwork`).

    Parameters
    depth_configs:
        List of hidden-layer configurations.

    Returns
    list[dict]
        One dict per configuration:
        ``{"hidden_layers", "depth", "train_f1", "test_f1"}``.
    """
    if depth_configs is None:
        depth_configs = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]

    print("\n" + "=" * 60)
    print("Experiment D: Varying depth — ReLU activation")
    print("=" * 60)

    n, r = X_train.shape[0], Y_train.shape[0]
    results: list[dict] = []

    for h in depth_configs:
        depth = len(h)
        print(f"\nTraining Model with Depth {depth}: {h}")
        model = GenericNeuralNetwork(
            n=n, hidden_layers=h, r=r,
            hidden_act=relu, hidden_act_deriv=relu_deriv,
            seed=cfg.seed,
        )
        model.train(X_train, Y_train, cfg.batch_size, cfg.epochs, cfg.lr, cfg.tol)

        true_test = np.argmax(Y_test, axis=0)
        precision, recall, f1_per_class, _ = precision_recall_fscore_support(
            true_test, model.predict(X_test), zero_division=0
        )
        print("  Class-wise Test Metrics (first 5 classes):")
        for c in range(min(5, r)):
            print(
                f"    Class {c:2d} -> Precision: {precision[c]:.4f}  "
                f"Recall: {recall[c]:.4f}  F1: {f1_per_class[c]:.4f}"
            )

        train_f1, test_f1 = _evaluate(model, X_train, Y_train, X_test, Y_test)
        print(f"  -> Train Avg F1: {train_f1:.4f} | Test Avg F1: {test_f1:.4f}")
        results.append({"hidden_layers": h, "depth": depth, "train_f1": train_f1, "test_f1": test_f1})

    # ---- Plot ----
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        depths = [r["depth"] for r in results]
        labels = [str(r["hidden_layers"]) for r in results]
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=depths, y=[r["test_f1"] for r in results], marker="o", label="Test F1")
        sns.lineplot(x=depths, y=[r["train_f1"] for r in results], marker="x", label="Train F1")
        plt.title("Part D: Average F1 Score vs. Network Depth (ReLU)")
        plt.xlabel("Depth (Number of Hidden Layers)")
        plt.ylabel("Average F1 Score (Macro)")
        plt.xticks(depths, labels=labels, rotation=45, ha="right")
        plt.grid(True, which="both", ls="--", c="0.7")
        plt.legend()
        plt.tight_layout()
        _maybe_show(cfg, "part_d_depth_relu.png")
    except ImportError:
        print("  [skip] matplotlib / seaborn not installed; skipping plot.")

    _save_csv(
        os.path.join(cfg.output_dir, "part_d_depth_relu.csv"),
        ["hidden_layers", "depth", "train_f1", "test_f1"],
        [
            [str(r["hidden_layers"]), r["depth"], f"{r['train_f1']:.6f}", f"{r['test_f1']:.6f}"]
            for r in results
        ],
    )

    return results


# Experiment E – sklearn MLPClassifier baseline

def run_mlp_baseline(
    X_train: NDArray[np.floating],
    Y_train: NDArray[np.floating],
    X_test: NDArray[np.floating],
    Y_test: NDArray[np.floating],
    cfg: ExperimentConfig,
    hidden_layer_options: list[list[int]] | None = None,
) -> list[dict]:
    """Run the sklearn MLPClassifier baseline over several architectures (Part E).

    Uses ``solver='sgd'``, ``activation='relu'``, ``batch_size=32``,
    ``learning_rate_init=0.001``, ``early_stopping=True``, and the remaining
    defaults from the notebook cell, to keep a fair comparison against the
    custom implementation.

    Parameters
    hidden_layer_options:
        List of hidden-layer configurations.
        ``[[512], [512,256], [512,256,128], [512,256,128,64]]``.

    Returns
    list[dict]
        One dict per configuration:
        ``{"hidden_layers", "accuracy", "train_f1", "test_f1"}``.
    """
    if hidden_layer_options is None:
        hidden_layer_options = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]

    print("\n" + "=" * 60)
    print("Experiment E: sklearn MLPClassifier baseline")
    print("=" * 60)

    # sklearn expects (n_samples, n_features) and integer labels
    X_train_sk = X_train.T
    X_test_sk = X_test.T
    y_train_1d = np.argmax(Y_train, axis=0)
    y_test_1d = np.argmax(Y_test, axis=0)

    results: list[dict] = []

    for h in hidden_layer_options:
        print(f"\nMLPClassifier hidden_layers={h}")
        mlp = MLPClassifier(
            hidden_layer_sizes=tuple(h),
            activation="relu",
            solver="sgd",
            alpha=0,
            batch_size=32,
            learning_rate="constant",
            learning_rate_init=0.001,
            max_iter=200,
            shuffle=True,
            tol=1e-4,
            verbose=False,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )
        mlp.fit(X_train_sk, y_train_1d)

        train_preds = mlp.predict(X_train_sk)
        test_preds = mlp.predict(X_test_sk)

        accuracy = float(mlp.score(X_test_sk, y_test_1d))
        train_f1 = float(f1_score(y_train_1d, train_preds, average="macro", zero_division=0))
        test_f1 = float(f1_score(y_test_1d, test_preds, average="macro", zero_division=0))

        print(f"  Architecture: {h} | Test Accuracy: {accuracy:.4f}")
        print(f"  -> Train Avg F1: {train_f1:.4f} | Test Avg F1: {test_f1:.4f}")
        results.append({"hidden_layers": h, "accuracy": accuracy, "train_f1": train_f1, "test_f1": test_f1})

    # ---- Plot ----
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        labels = [str(r["hidden_layers"]) for r in results]
        x = list(range(len(results)))
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=x, y=[r["test_f1"] for r in results], marker="o", label="Test F1")
        sns.lineplot(x=x, y=[r["train_f1"] for r in results], marker="x", label="Train F1")
        plt.title("Part E: sklearn MLPClassifier — F1 vs. Architecture")
        plt.xlabel("Hidden Layer Architecture")
        plt.ylabel("Average F1 Score (Macro)")
        plt.xticks(x, labels=labels, rotation=45, ha="right")
        plt.grid(True, which="both", ls="--", c="0.7")
        plt.legend()
        plt.tight_layout()
        _maybe_show(cfg, "part_e_mlp_baseline.png")
    except ImportError:
        print("  [skip] matplotlib / seaborn not installed; skipping plot.")

    _save_csv(
        os.path.join(cfg.output_dir, "part_e_mlp_baseline.csv"),
        ["hidden_layers", "accuracy", "train_f1", "test_f1"],
        [
            [str(r["hidden_layers"]), f"{r['accuracy']:.6f}", f"{r['train_f1']:.6f}", f"{r['test_f1']:.6f}"]
            for r in results
        ],
    )

    return results


# Notebook 02 – scratch vs. transfer learning (digits dataset)

def run_scratch_digits(
    X_train: NDArray[np.floating],
    Y_train: NDArray[np.floating],
    X_test: NDArray[np.floating],
    Y_test: NDArray[np.floating],
    cfg: ExperimentConfig,
) -> tuple[list[float], list[float]]:
    """Train [512,256,128,64] ReLU network from scratch on the digits dataset."""
    print("Experiment: Scratch model on digits [512,256,128,64]")
    n, r = X_train.shape[0], Y_train.shape[0]
    model = GenericNeuralNetwork(
        n=n, hidden_layers=[512, 256, 128, 64], r=r,
        hidden_act=relu, hidden_act_deriv=relu_deriv, seed=cfg.seed,
    )

    train_f1_history: list[float] = []
    test_f1_history: list[float] = []

    for epoch in range(cfg.epochs):
        model.train(X_train, Y_train, cfg.batch_size, epochs=1, lr=cfg.lr, tol=cfg.tol)
        train_f1 = float(f1_score(np.argmax(Y_train, axis=0), model.predict(X_train), average="macro", zero_division=0))
        test_f1  = float(f1_score(np.argmax(Y_test,  axis=0), model.predict(X_test),  average="macro", zero_division=0))
        train_f1_history.append(train_f1)
        test_f1_history.append(test_f1)
        print(f"  Epoch {epoch+1}/{cfg.epochs} | Train F1: {train_f1:.4f} | Test F1: {test_f1:.4f}")

    print(f"Final => Train F1: {train_f1_history[-1]:.4f} | Test F1: {test_f1_history[-1]:.4f}")
    return train_f1_history, test_f1_history


def run_transfer_learning(
    X_train: NDArray[np.floating],
    Y_train: NDArray[np.floating],
    X_test: NDArray[np.floating],
    Y_test: NDArray[np.floating],
    cfg: ExperimentConfig,
    pretrained_model_path: str,
) -> tuple[list[float], list[float]]:
    """Fine-tune the pretrained consonant model on the digits dataset.

    Replaces the final weight/bias layer with a randomly initialised
    10-output layer and trains for cfg.epochs epochs.
    """
    print(f"Experiment: Transfer learning from {pretrained_model_path}")
    with open(pretrained_model_path, "rb") as fh:
        payload = pickle.load(fh)
    model: GenericNeuralNetwork = payload["model"] if isinstance(payload, dict) else payload

    # Swap final layer to match 10 digit classes
    r_new = Y_train.shape[0]
    last_layer = model.L
    fan_in = model.W[last_layer].shape[1]
    np.random.seed(cfg.seed)
    model.W[last_layer] = np.random.randn(r_new, fan_in) * 0.01
    model.b[last_layer] = np.zeros((r_new, 1))

    train_f1_history: list[float] = []
    test_f1_history: list[float] = []

    for epoch in range(cfg.epochs):
        model.train(X_train, Y_train, cfg.batch_size, epochs=1, lr=cfg.lr, tol=cfg.tol)
        train_f1 = float(f1_score(np.argmax(Y_train, axis=0), model.predict(X_train), average="macro", zero_division=0))
        test_f1  = float(f1_score(np.argmax(Y_test,  axis=0), model.predict(X_test),  average="macro", zero_division=0))
        train_f1_history.append(train_f1)
        test_f1_history.append(test_f1)
        print(f"  Epoch {epoch+1}/{cfg.epochs} | Train F1: {train_f1:.4f} | Test F1: {test_f1:.4f}")

    print(f"Final => Train F1: {train_f1_history[-1]:.4f} | Test F1: {test_f1_history[-1]:.4f}")
    return train_f1_history, test_f1_history


def run_transfer_comparison(
    X_train: NDArray[np.floating],
    Y_train: NDArray[np.floating],
    X_test: NDArray[np.floating],
    Y_test: NDArray[np.floating],
    cfg: ExperimentConfig,
    pretrained_model_path: str,
) -> None:
    """Run both scratch and transfer experiments and plot/save the comparison."""
    scratch_train, scratch_test = run_scratch_digits(X_train, Y_train, X_test, Y_test, cfg)
    transfer_train, transfer_test = run_transfer_learning(X_train, Y_train, X_test, Y_test, cfg, pretrained_model_path)

    epochs = list(range(1, cfg.epochs + 1))
    rows = [
        [e, f"{st:.6f}", f"{ss:.6f}", f"{tt:.6f}", f"{ts:.6f}"]
        for e, st, ss, tt, ts in zip(epochs, scratch_train, scratch_test, transfer_train, transfer_test)
    ]
    _save_csv(
        os.path.join(cfg.output_dir, "transfer_comparison.csv"),
        ["epoch", "scratch_train_f1", "scratch_test_f1", "transfer_train_f1", "transfer_test_f1"],
        rows,
    )

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, scratch_train,  label="Scratch Train F1",   linestyle="--")
        plt.plot(epochs, scratch_test,   label="Scratch Test F1",    linestyle="-")
        plt.plot(epochs, transfer_train, label="Transfer Train F1",  linestyle=":")
        plt.plot(epochs, transfer_test,  label="Transfer Test F1",   linestyle="-.")
        plt.xlabel("Epoch")
        plt.ylabel("F1-score (Macro)")
        plt.title("Scratch vs. Transfer Learning — F1 per Epoch")
        plt.xticks(epochs)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        _maybe_show(cfg, "transfer_comparison.png")
    except ImportError:
        print("  [skip] matplotlib not installed; skipping plot.")


def _load_data(
    train_dir: str,
    test_dir: str,
    img_height: int = 32,
    img_width: int = 32,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    list[str],
]:
    X_train_raw, Y_train_raw, class_names = load_devanagari_data(
        train_dir, img_height=img_height, img_width=img_width
    )
    X_test, Y_test, _ = load_devanagari_data(
        test_dir, img_height=img_height, img_width=img_width
    )
    X_train, Y_train = shuffle_data(X_train_raw, Y_train_raw)
    return X_train, Y_train, X_test, Y_test, class_names


# argparse CLI

def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the top-level argument parser."""
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--train-dir", required=True, metavar="DIR",
                        help="Root directory of training images (class-per-folder).")
    parent.add_argument("--test-dir", required=True, metavar="DIR",
                        help="Root directory of test images.")
    parent.add_argument("--img-height", type=int, default=32, metavar="H",
                        help="Resize height in pixels (default: 32).")
    parent.add_argument("--img-width", type=int, default=32, metavar="W",
                        help="Resize width in pixels (default: 32).")
    parent.add_argument("--epochs", type=int, default=50, metavar="N",
                        help="Max training epochs (default: 50).")
    parent.add_argument("--batch-size", type=int, default=32, metavar="B",
                        help="Mini-batch size (default: 32).")
    parent.add_argument("--lr", type=float, default=0.01, metavar="LR",
                        help="Learning rate (default: 0.01).")
    parent.add_argument("--tol", type=float, default=1e-4, metavar="TOL",
                        help="Early-stopping tolerance (default: 1e-4).")
    parent.add_argument("--seed", type=int, default=42, metavar="S",
                        help="Random seed (default: 42).")
    parent.add_argument("--output-dir", default="results", metavar="DIR",
                        help="Directory for CSV and PNG outputs (default: results/).")
    parent.add_argument("--save-model", default=None, metavar="PATH",
                        help="Pickle the best/final model to this path.")
    parent.add_argument("--no-plots", action="store_true",
                        help="Generate plots but do not display them (headless mode).")

    parser = argparse.ArgumentParser(
        prog="experiments",
        description="Rerun all notebook experiment loops from the command line.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("all", parents=[parent],
                   help="Run every experiment in sequence.")
    sub.add_parser("hidden-units", parents=[parent],
                   help="Part B: sweep over single-layer widths (sigmoid).")
    sub.add_parser("depth", parents=[parent],
                   help="Part C: sweep over depth with sigmoid activation.")
    sub.add_parser("activation", parents=[parent],
                   help="Part D: sweep over depth with ReLU activation.")
    sub.add_parser("mlp", parents=[parent],
                   help="Part E: sklearn MLPClassifier baseline.")

    # transfer subcommand needs an extra --pretrained-model flag
    transfer_parent = argparse.ArgumentParser(add_help=False, parents=[parent])
    transfer_parent.add_argument(
        "--pretrained-model",
        default="model/pretrained_consonant_model.pkl",
        metavar="PATH",
        help="Pretrained consonant model pickle (default: model/pretrained_consonant_model.pkl).",
    )
    sub.add_parser("transfer", parents=[transfer_parent],
                   help="Notebook 02: scratch vs. transfer learning on digits.")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and dispatch to the appropriate experiment(s).

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]`` when ``None``).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = ExperimentConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        tol=args.tol,
        seed=args.seed,
        output_dir=args.output_dir,
        save_model=args.save_model,
        no_plots=args.no_plots,
    )
    _ensure_dir(cfg.output_dir)

    print(f"Loading data …  train={args.train_dir}  test={args.test_dir}")
    X_train, Y_train, X_test, Y_test, class_names = _load_data(
        args.train_dir, args.test_dir,
        img_height=args.img_height, img_width=args.img_width,
    )
    print(f"Classes ({len(class_names)}): {class_names[:5]} …")

    cmd = args.command

    if cmd in ("all", "hidden-units"):
        run_hidden_units(X_train, Y_train, X_test, Y_test, cfg)

    if cmd in ("all", "depth"):
        run_depth_sigmoid(X_train, Y_train, X_test, Y_test, cfg)

    if cmd in ("all", "activation"):
        run_depth_relu(X_train, Y_train, X_test, Y_test, cfg)

    if cmd in ("all", "mlp"):
        run_mlp_baseline(X_train, Y_train, X_test, Y_test, cfg)

    if cmd == "transfer":
        run_transfer_comparison(
            X_train, Y_train, X_test, Y_test, cfg,
            pretrained_model_path=args.pretrained_model,
        )

    # Optional: save the best custom model (ReLU [512,256,128,64])
    if cfg.save_model and cmd == "all":
        n, r = X_train.shape[0], Y_train.shape[0]
        print(f"\nTraining final model [512,256,128,64] for export → {cfg.save_model}")
        best = GenericNeuralNetwork(
            n=n, hidden_layers=[512, 256, 128, 64], r=r,
            hidden_act=relu, hidden_act_deriv=relu_deriv,
            seed=cfg.seed,
        )
        best.train(X_train, Y_train, cfg.batch_size, cfg.epochs, cfg.lr, cfg.tol)
        model_data = {"model": best, "class_names": class_names}
        with open(cfg.save_model, "wb") as fh:
            pickle.dump(model_data, fh)
        print(f"  [saved] {cfg.save_model}")

    print("\nAll requested experiments complete.")


if __name__ == "__main__":
    main()
