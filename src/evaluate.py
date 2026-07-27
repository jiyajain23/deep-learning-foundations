"""
evaluate.py
===========
Standalone inference script for the Devanagari handwritten character
classifier.

Given a pickled model file and an image path, this module will:

1. Load the ``GenericNeuralNetwork`` (or any ``sklearn``-compatible estimator)
   from a pickle file produced by :mod:`experiments` or the training notebook.
2. Pre-process the image to match training conditions (grayscale, 32×32 resize,
   flatten, normalise to ``[0, 1]``).
3. Print—and return—the predicted class name.

Command-line usage
::

    python src/evaluate.py path/to/model.pkl path/to/image.png

    # Override the default 32×32 resolution if your model used a different size:
    python src/evaluate.py path/to/model.pkl path/to/image.png --img-height 64 --img-width 64

    # Print the full softmax probability vector alongside the prediction:
    python src/evaluate.py path/to/model.pkl path/to/image.png --show-probs

Programmatic usage
::

    from src.evaluate import predict_single

    label, confidence = predict_single(
        model_path="model/pretrained_consonant_model.pkl",
        image_path="path/to/cha.png",
    )
    print(f"Predicted: {label}  (confidence {confidence:.1%})")

    # Request the full probability vector in one call:
    label, confidence, probs = predict_single(
        model_path="model/pretrained_consonant_model.pkl",
        image_path="path/to/cha.png",
        return_probs=True,
    )

"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def preprocess_image(
    image_path: str,
    img_height: int = 32,
    img_width: int = 32,
) -> NDArray[np.floating]:
    """Load, convert to grayscale, resize, flatten, and normalize an image."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path!r}")

    img = Image.open(image_path).convert("L").resize((img_width, img_height))
    arr: NDArray[np.floating] = np.array(img, dtype=float).flatten() / 255.0
    return arr.reshape(-1, 1)  # shape: (n_features, 1)


def _load_model_and_classes(
    model_path: str,
    data_dir: str | None = None,
) -> tuple[Any, list[str]]:
    """Deserialize model from a pickle file and resolve class names."""
    try:
        with open(model_path, "rb") as fh:
            payload = pickle.load(fh)
    except (pickle.UnpicklingError, Exception) as exc:
        raise RuntimeError(
            f"Could not load model from {model_path!r}. "
            "The file may be corrupted or was pickled with an incompatible library version."
        ) from exc

    # Dict layout from experiments.py --save-model
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        class_names: list[str] = payload.get("class_names", [])
    else:
        model = payload
        class_names = []

    if not class_names and data_dir is not None and os.path.isdir(data_dir):
        class_names = sorted(
            d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
        )

    if not class_names:
        try:
            n_classes: int = model.layer_sizes[-1]
        except AttributeError:
            try:
                n_classes = len(model.classes_)
            except AttributeError:
                n_classes = 0
        class_names = [f"class_{i}" for i in range(n_classes)]
        warnings.warn(
            f"No real class names found for {model_path!r}. "
            f"Using generic placeholder labels (class_0 … class_{n_classes - 1}). "
            "Pass --data-dir to resolve actual class names.",
            UserWarning,
            stacklevel=2,
        )

    return model, class_names


def predict_single(
    model_path: str,
    image_path: str,
    img_height: int = 32,
    img_width: int = 32,
    data_dir: str | None = None,
    return_probs: bool = False,
) -> tuple[str, float] | tuple[str, float, NDArray[np.floating]]:
    """Run inference on a single image and return the predicted class label.

    This is the primary public API of this module.

    The full probability vector is computed in every call (it is a byproduct
    of the forward pass) and is returned when *return_probs* is ``True``,
    avoiding any need to re-run inference just to retrieve it.

    Returns
    -------
    predicted_class : str
        Human-readable class name (e.g. ``"character_1_ka"``).
    confidence : float
        Softmax probability of the predicted class in ``[0, 1]``, or
        ``float('nan')`` when the model cannot produce probabilities.
    probs : NDArray[np.floating]
        Only present when *return_probs* is ``True``.  Full probability
        vector of shape ``(r,)``, or an empty array when unavailable.

    """
    model, class_names = _load_model_and_classes(model_path, data_dir)
    X = preprocess_image(image_path, img_height=img_height, img_width=img_width)

    probs: NDArray[np.floating] = np.array([])

    # -----------------------------------------------------------------
    if hasattr(model, "forward"):
        activations = model.forward(X)          # column-vector input
        probs = activations[-1].flatten()        # shape (r,)
        pred_idx: int = int(np.argmax(probs))
        confidence: float = float(probs[pred_idx])

    elif hasattr(model, "predict_proba"):
        X_sk = X.T                              # sklearn expects (1, n_features)
        probs = model.predict_proba(X_sk)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
    elif hasattr(model, "predict"):
        X_sk = X.T
        pred_idx = int(model.predict(X_sk)[0])
        confidence = float("nan")
        # probs stays as np.array([]) — no probability estimate available
    else:
        raise ValueError(
            f"The loaded object ({type(model).__name__}) does not expose "
            "a recognised .forward(), .predict_proba(), or .predict() method."
        )

    predicted_class = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
    if return_probs:
        return predicted_class, confidence, probs
    return predicted_class, confidence



def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the argument parser for the CLI entry-point."""
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description=(
            "Load a pickled Devanagari classifier and predict the class of "
            "a single image."
        ),
    )
    parser.add_argument("model_path", metavar="MODEL",
                        help="Path to the pickled model (.pkl).")
    parser.add_argument("image_path", metavar="IMAGE",
                        help="Path to the image file to classify.")
    parser.add_argument("--img-height", type=int, default=32, metavar="H",
                        help="Resize height in pixels — must match training (default: 32).")
    parser.add_argument("--img-width", type=int, default=32, metavar="W",
                        help="Resize width in pixels — must match training (default: 32).")
    parser.add_argument("--data-dir", default=None, metavar="DIR",
                        help=(
                            "Dataset root directory.  Used to derive class names "
                            "when the pickle was saved without them."
                        ))
    parser.add_argument("--show-probs", action="store_true",
                        help="Print the top-k class probabilities.")
    parser.add_argument("--top-k", type=int, default=10, metavar="K",
                        help="Number of top classes to show with --show-probs (default: 10).")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run inference on a single image.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]`` when ``None``).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Always request probs so --show-probs doesn't need a second forward pass.
    predicted_class, confidence, probs = predict_single(
        model_path=args.model_path,
        image_path=args.image_path,
        img_height=args.img_height,
        img_width=args.img_width,
        data_dir=args.data_dir,
        return_probs=True,
    )

    conf_str = f"{confidence:.1%}" if not np.isnan(confidence) else "n/a"

    if not args.show_probs:
        print(f"Predicted class : {predicted_class}")
        print(f"Confidence      : {conf_str}")
    else:
        print(f"Predicted class : {predicted_class}  (confidence {conf_str})\n")
        if probs.size > 0:
            _, class_names = _load_model_and_classes(args.model_path, args.data_dir)
            print("Full probability distribution:")
            sorted_idx = np.argsort(probs)[::-1]
            for rank, idx in enumerate(sorted_idx[:args.top_k], start=1):
                name = class_names[idx] if idx < len(class_names) else str(idx)
                print(f"  {rank:2d}. {name:<35s} {probs[idx]:.4f}")


if __name__ == "__main__":
    main()
