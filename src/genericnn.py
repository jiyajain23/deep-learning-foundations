import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

def sigmoid_deriv(a):
    return a * (1.0 - a)

def relu(z):
    return np.maximum(0, z)

def relu_deriv(a):
    return (a > 0).astype(float)

class GenericNeuralNetwork:
    def __init__(self, n, hidden_layers, r, hidden_act=sigmoid, hidden_act_deriv=sigmoid_deriv, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.hidden_act = hidden_act
        self.hidden_act_deriv = hidden_act_deriv
        self.layer_sizes = [n] + list(hidden_layers) + [r]
        self.L = len(self.layer_sizes) - 1

        self.W, self.b = {}, {}
        for l in range(1, self.L + 1):
            fan_in, fan_out = self.layer_sizes[l - 1], self.layer_sizes[l]
            scale = np.sqrt(2.0 / fan_in) if hidden_act == relu and l < self.L else np.sqrt(1.0 / fan_in)
            self.W[l] = np.random.randn(fan_out, fan_in) * scale
            self.b[l] = np.zeros((fan_out, 1))

    def forward(self, X):
        activations = [X]
        a = X

        for l in range(1, self.L + 1):
            z = self.W[l] @ a + self.b[l]

            if l < self.L:
                a = self.hidden_act(z)
            else:
                exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
                a = exp_z / np.sum(exp_z, axis=0, keepdims=True)

            activations.append(a)

        return activations

    def backward(self, activations, Y):
        M = Y.shape[1]
        grads_W, grads_b = {}, {}
        delta = activations[self.L] - Y

        for l in range(self.L, 0, -1):
            a_prev = activations[l - 1]
            grads_W[l] = (delta @ a_prev.T) / M
            grads_b[l] = np.mean(delta, axis=1, keepdims=True)

            if l > 1:
                delta = (self.W[l].T @ delta) * self.hidden_act_deriv(a_prev)

        return grads_W, grads_b

    def predict(self, X):
        activations = self.forward(X)
        return np.argmax(activations[-1], axis=0)

    def train(self, X, Y, M_batch_size, epochs, lr, tol=1e-4):
        m_total = X.shape[1]
        prev_train_acc = None
        history = []

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

            train_acc = np.mean(self.predict(X) == np.argmax(Y, axis=0))
            history.append(train_acc)
            print(f"Epoch {epoch}/{epochs} | Train Accuracy: {train_acc:.4f}")

            if prev_train_acc is not None and abs(train_acc - prev_train_acc) < tol:
                print(f"Early stopping at epoch {epoch} as accuracy difference ({abs(train_acc - prev_train_acc):.4f}) is less than tolerance ({tol}).")
                break
            prev_train_acc = train_acc

        return history
