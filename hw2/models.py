import numpy as np


class Linear:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.biases = np.zeros(output_dim)
        self.input = None
        self.d_weights = None
        self.d_biases = None

    def forward(self, input_data):
        self.input = input_data
        self.input = self.input.reshape(self.input.shape[0], -1)
        return np.dot(self.input, self.weights) + self.biases

    def backward(self, output_gradient):
        input_gradient = np.dot(output_gradient, self.weights.T)
        self.d_weights = np.dot(self.input.T, output_gradient)
        self.d_biases = np.sum(output_gradient, axis=0)
        return input_gradient


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, output_gradient):
        input_gradient = (self.input > 0) * output_gradient
        return input_gradient


class Softmax:
    def __init__(self):
        self.out = None

    def forward(self, x):
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.out = exps / np.sum(exps, axis=-1, keepdims=True)
        return self.out

    def backward(self, d_out, Y_true):
        batch_size = d_out.shape[0]
        d_input = self.out - Y_true
        d_input /= batch_size
        return d_input


class MSE:
    def forward(self, Y_pred, Y_true):
        self.Y_pred = Y_pred
        self.Y_true = Y_true
        return np.mean(np.square(Y_true - Y_pred))

    def backward(self):
        return 2 * (self.Y_pred - self.Y_true) / self.Y_pred.size


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.loss = None
        self.loss_history = []

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, Y_true):
        dY = self.loss.backward()
        for layer in reversed(self.layers):
            if isinstance(layer, Softmax):
                dY = layer.backward(dY, Y_true)
            else:
                dY = layer.backward(dY)

    def compute_loss(self, Y_pred, Y_true):
        return self.loss.forward(Y_pred, Y_true)

    def update_weights(self, learning_rate):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.weights -= learning_rate * layer.d_weights
                layer.biases -= learning_rate * layer.d_biases

    def fit(self, X, Y, epochs, batch_size, learning_rate):
        for epoch in range(epochs):
            idx = np.random.permutation(X.shape[0])
            X = X[idx]
            Y = Y[idx]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                Y_batch = Y[i:i+batch_size]

                Y_pred = self.forward(X_batch)
                loss = self.compute_loss(Y_pred, Y_batch)
                self.backward(Y_batch)
                self.update_weights(learning_rate)

                self.loss_history.append(loss)

            if (epoch+1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}')

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

