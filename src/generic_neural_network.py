"""
generic_neural_network.py

A from-scratch, NumPy-only implementation of a fully-connected (dense)
feed-forward neural network that supports arbitrary depth and two built-in
activation functions (sigmoid and ReLU).

Design notes
------------
* Weights are stored in 1-indexed dicts ``W`` and ``b`` so that layer *l*
  always corresponds to ``W[l]`` / ``b[l]`` — matching the mathematical
  notation commonly used in lecture notes.
* The output layer always uses the **softmax** activation, making the class
  suitable for multi-class classification out of the box.
* Initialisation follows He (ReLU hidden layers) or Glorot (sigmoid / output
  layer) scaling to keep activations well-conditioned from the first forward
  pass.
"""

from __future__ import annotations

from typing import Callable
import numpy as np
from numpy.typing import NDArray

# Activation functions and their derivatives

def sigmoid(z: NDArray[np.floating]) -> NDArray[np.floating]:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))


def sigmoid_deriv(a: NDArray[np.floating]) -> NDArray[np.floating]:
    return a * (1.0 - a)


def relu(z: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.maximum(0, z)


def relu_deriv(a: NDArray[np.floating]) -> NDArray[np.floating]:
    return (a > 0).astype(float)


# Neural network class

class GenericNeuralNetwork:
    """Fully-connected feed-forward neural network for multi-class classification.

    Architecture
    Input layer — n features.
    Hidden layers — configurable depth and width; activation function
    is supplied by the caller.
    Output layer — r units with softmax activation; a single call to
    predict() returns the argmax class index.

    Training uses mini-batch gradient descent (SGD) with cross-entropy
    loss, implemented via the standard forward / backward pass pair.

    Parameters
    n:
        Number of input features (dimensionality of each sample).
    hidden_layers:
        Sequence of integers specifying the number of units in each hidden
        layer.  E.g. ``[128, 64]`` creates two hidden layers.
    r:
        Number of output classes.
    hidden_act:
        Activation function applied to every hidden layer.  Must accept and
        return an ``NDArray`` of the same shape.  Defaults to
        :func:`sigmoid`.
    hidden_act_deriv:
        Derivative of *hidden_act*, expressed in terms of the post-activation
        value (not the pre-activation *z*).  Must match *hidden_act*.
        Defaults to :func:`sigmoid_deriv`.
    seed:
        Optional integer seed passed to numpy.random.seed for
        reproducibility.  When None, the global NumPy RNG state is used.

    """

    def __init__(
        self,
        n: int,
        hidden_layers: list[int] | tuple[int, ...],
        r: int,
        hidden_act: Callable[[NDArray[np.floating]], NDArray[np.floating]] = sigmoid,
        hidden_act_deriv: Callable[[NDArray[np.floating]], NDArray[np.floating]] = sigmoid_deriv,
        seed: int | None = None,
        l2_lambda: float = 0.0,
    ) -> None:
        if seed is not None:
            np.random.seed(seed)

        self.hidden_act = hidden_act
        self.hidden_act_deriv = hidden_act_deriv
        self.l2_lambda = l2_lambda
        self.layer_sizes: list[int] = [n] + list(hidden_layers) + [r]
        self.L: int = len(self.layer_sizes) - 1

        self.W: dict[int, NDArray[np.floating]] = {}
        self.b: dict[int, NDArray[np.floating]] = {}

        for l in range(1, self.L + 1):
            fan_in, fan_out = self.layer_sizes[l - 1], self.layer_sizes[l]
            scale = (
                np.sqrt(2.0 / fan_in)
                if hidden_act == relu and l < self.L
                else np.sqrt(1.0 / fan_in)
            )
            self.W[l] = np.random.randn(fan_out, fan_in) * scale
            self.b[l] = np.zeros((fan_out, 1))

    # Forward pass

    def forward(
        self, X: NDArray[np.floating]
    ) -> list[NDArray[np.floating]]:
        """Run a full forward pass and return all layer activations.

        Parameters
        X:
            Input matrix of shape (n, m) where n is the feature
            dimensionality and m is the batch size.

        Returns
        list[NDArray[np.floating]]
            A list of L + 1 arrays.  Index 0 is the input X itself;
            index l (for l in 1..L) is the post-activation output of
            layer l.  The final element is the softmax probability matrix
            of shape (r, m).
        """
        activations: list[NDArray[np.floating]] = [X]
        a: NDArray[np.floating] = X

        for l in range(1, self.L + 1):
            z: NDArray[np.floating] = self.W[l] @ a + self.b[l]

            if l < self.L:
                a = self.hidden_act(z)
            else:
                # Numerically stable softmax
                exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
                a = exp_z / np.sum(exp_z, axis=0, keepdims=True)

            activations.append(a)

        return activations

    def backward(
        self,
        activations: list[NDArray[np.floating]],
        Y: NDArray[np.floating],
    ) -> tuple[dict[int, NDArray[np.floating]], dict[int, NDArray[np.floating]]]:
        """Compute gradients via back-propagation with optional L2 regularization."""
        M: int = Y.shape[1]
        grads_W: dict[int, NDArray[np.floating]] = {}
        grads_b: dict[int, NDArray[np.floating]] = {}
        delta: NDArray[np.floating] = activations[self.L] - Y

        for l in range(self.L, 0, -1):
            a_prev = activations[l - 1]
            # L2 penalty: adds lambda * W to the weight gradient (biases not regularized)
            grads_W[l] = (delta @ a_prev.T) / M + self.l2_lambda * self.W[l]
            grads_b[l] = np.mean(delta, axis=1, keepdims=True)

            if l > 1:
                delta = (self.W[l].T @ delta) * self.hidden_act_deriv(a_prev)

        return grads_W, grads_b

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.intp]:
        activations = self.forward(X)
        return np.argmax(activations[-1], axis=0)

    def train(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
        M_batch_size: int,
        epochs: int,
        lr: float,
        tol: float = 1e-4,
    ) -> list[float]:
        """Train with mini-batch SGD; returns per-epoch train accuracy history."""
        m_total: int = X.shape[1]
        prev_train_acc: float | None = None
        history: list[float] = []

        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(m_total)
            X_shuf, Y_shuf = X[:, perm], Y[:, perm]

            for start in range(0, m_total, M_batch_size):
                end = start + M_batch_size
                X_batch = X_shuf[:, start:end]
                Y_batch = Y_shuf[:, start:end]

                activations = self.forward(X_batch)
                grads_W, grads_b = self.backward(activations, Y_batch)

                for l in range(1, self.L + 1):
                    self.W[l] -= lr * grads_W[l]
                    self.b[l] -= lr * grads_b[l]

            # O(m_total) forward pass every epoch — for large datasets consider
            # replacing with a random subsample (e.g. X[:, :2000]) to stay fast.
            train_acc: float = float(
                np.mean(self.predict(X) == np.argmax(Y, axis=0))
            )
            history.append(train_acc)
            print(f"Epoch {epoch}/{epochs} | Train Accuracy: {train_acc:.4f}")

            if prev_train_acc is not None and abs(train_acc - prev_train_acc) < tol:
                print(
                    f"Early stopping at epoch {epoch} as accuracy difference "
                    f"({abs(train_acc - prev_train_acc):.4f}) is less than "
                    f"tolerance ({tol})."
                )
                break
            prev_train_acc = train_acc

        return history
