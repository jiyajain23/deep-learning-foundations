"""
data_utils.py
=============
Utility functions for loading and preprocessing image datasets stored in a
standard **class-per-folder** directory layout.

Expected directory structure
----------------------------
::

    data_dir/
    ├── class_a/
    │   ├── image_001.png
    │   └── ...
    ├── class_b/
    │   └── ...
    └── ...

The functions here were originally written for the
`Devanagari Handwritten Character Dataset
<https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset>`_
but work with any image classification dataset that follows the same layout.
"""

from __future__ import annotations

import os

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def load_devanagari_data(
    data_dir: str,
    img_height: int = 32,
    img_width: int = 32,
) -> tuple[NDArray[np.floating], NDArray[np.floating], list[str]]:
    """Load a class-per-folder image dataset into NumPy arrays.

    Each image is:

    1. Opened with Pillow and converted to **grayscale** (``'L'`` mode).
    2. Resized to ``(img_width, img_height)`` pixels.
    3. Flattened to a 1-D vector of length ``img_height * img_width``.
    4. Normalised to the ``[0, 1]`` range by dividing by 255.

    Parameters
    data_dir:
        Path to the root directory containing one sub-directory per class.
    img_height:
        Height (in pixels) to which every image is resized.  Defaults to 32.
    img_width:
        Width (in pixels) to which every image is resized.  Defaults to 32.

    Returns
    X : NDArray[np.floating]
        Feature matrix of shape ``(img_height * img_width, m)`` where *m* is
        the total number of successfully loaded images.  Each column is one
        flattened, normalised image.
    Y_one_hot : NDArray[np.floating]
        One-hot label matrix of shape ``(num_classes, m)``.
        ``Y_one_hot[c, i] == 1`` iff sample *i* belongs to class *c*.
    classes : list[str]
        Sorted list of class names (sub-directory names), where index *c*
        corresponds to row *c* of *Y_one_hot*.
    """
    classes: list[str] = sorted(os.listdir(data_dir))
    num_classes: int = len(classes)

    X_list: list[NDArray[np.floating]] = []
    Y_list: list[int] = []

    print(f"Loading data from {data_dir}")

    for class_idx, class_name in enumerate(classes):
        class_path: str = os.path.join(data_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path: str = os.path.join(class_path, img_name)

            try:
                img = Image.open(img_path).convert("L").resize((img_width, img_height))
                img_array: NDArray[np.floating] = np.array(img).flatten()
                img_array = img_array / 255.0
                X_list.append(img_array)
                Y_list.append(class_idx)
            except Exception as e:  # noqa: BLE001
                print(f"Skipping {img_path}: {e}")

    X: NDArray[np.floating] = np.array(X_list).T
    m: int = len(Y_list)
    Y: NDArray[np.intp] = np.array(Y_list)
    Y_one_hot: NDArray[np.floating] = np.zeros((num_classes, m))
    Y_one_hot[Y, np.arange(m)] = 1

    print(f"Loaded {m} images. X shape: {X.shape}, Y shape: {Y_one_hot.shape}")

    return X, Y_one_hot, classes


def shuffle_data(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    np.random.seed(42)
    m: int = X.shape[1]
    permutation: NDArray[np.intp] = np.random.permutation(m)
    return X[:, permutation], Y[:, permutation]
