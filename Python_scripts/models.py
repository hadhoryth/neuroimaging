import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers
        self.parameters = dict()

    def add_hidden(self, n_nodes):
        self.layers = np.append(self.layers, [n_nodes])

    def _init_weights(self):
        self.parameters = dict()
        np.random.seed(1)
        for l in range(1, len(self.layers)):
            w, b = (self.layers[l], self.layers[l - 1]), (self.layers[l], 1)
            self.parameters['W' + str(l)] = np.random.randn(w[0], w[1]) * 0.01
            self.parameters['b' + str(l)] = np.zeros(b)

        # self.parameters['W1'] = np.array(
        #     [[0.1, 0.5], [0.4, 0.01], [0.08, 0.015]])
        # self.parameters['b1'] = np.zeros((3, 1))

        # self.parameters['W2'] = np.array(
        #     [[0.01, 0.03, 0.05], [0.07, 0.06, 0.05]])
        # self.parameters['b2'] = np.zeros((2, 1))

        # self.parameters['W3'] = np.array([0.001, 0.007]).reshape((1, 2))
        # self.parameters['b3'] = np.zeros((1, 1))

    def _forward_propagate(self, X):
        def activation(w, x, b, func):
            Z = np.dot(w, x) + b
            A = None
            if func == 'sigmoid':
                A = 1 / (1 + np.exp(-Z))
            elif func == 'relu':
                A = np.maximum(0, Z)
            return A, (A, Z)

        activation_func, A = 'relu', X
        activation_cache = [[A]]
        for l in range(1, len(self.layers)):
            A_prev = A
            w, b = self.parameters['W' + str(l)], self.parameters['b' + str(l)]
            if l == (len(self.layers) - 1):
                activation_func = 'sigmoid'
            A, cache = activation(w, A_prev, b, activation_func)
            activation_cache.append(cache)

        return activation_cache

    def _back_propagate(self, forward_cache, Y):
        def back_activation(dA, Z, W, A_prev, m, func):
            if func == 'sigmoid':
                s = 1 / (1 + np.exp(-Z))
                dZ = dA * s * (1 - s)
            elif func == 'relu':
                dZ = np.array(dA, copy=True)
                dZ[Z <= 0] = 0

            dW = 1 / m * np.dot(dZ, A_prev.T)
            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.dot(W.T, dZ)

            return dW, db, dA_prev

        grads = dict()
        for l in reversed(range(1, len(self.layers))):
            A_prev, Z = forward_cache[l - 1][0], forward_cache[l][1]
            W = self.parameters['W' + str(l)]
            if l == len(self.layers) - 1:
                A = forward_cache[l][0]
                dA = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
                activation_func = 'sigmoid'
            else:
                dA = grads['dA' + str(l + 1)]
                activation_func = 'relu'

            grads['dW' + str(l)], grads['db' + str(l)], grads['dA' + str(l)] \
                = back_activation(dA, Z, W, A_prev, Y.shape[1], activation_func)
            # print('W: ', W.shape, '  dW: ', grads['dW' + str(l)].shape)

        return grads

    def _update_params(self, grads, learning_rate):
        for l in range(1, len(self.layers)):
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - \
                learning_rate * grads["dW" + str(l)]
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - \
                learning_rate * grads["db" + str(l)]

    def _print_weights(self):
        print('W1: \n', self.parameters['W1'],
              ', b1: \n', self.parameters['b1'])
        print('W2: \n', self.parameters['W2'],
              ', b2: \n', self.parameters['b2'])
        print('W3: \n', self.parameters['W3'],
              ', b3: \n', self.parameters['b3'])

    def run(self, X, Y, learning_rate, num_iterations, print_cost=True):
        self._init_weights()
        self._print_weights()
        costs = []
        for i in range(num_iterations):
            forward_cache = self._forward_propagate(X)

            A_L = forward_cache[-1][0]
            cost = -1 / len(Y) * (np.dot(Y, np.log(A_L).T) +
                                  np.dot((1 - Y), np.log(1 - A_L).T))

            gradients = self._back_propagate(forward_cache, Y)
            self._update_params(gradients, learning_rate)
            costs.append(cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        return costs

    def predict(self, X):
        activations = self._forward_propagate(X)
        return activations[-1][0]


if __name__ == '__main__':
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
    print(X)
    Y = np.array([1, 0, 0, 1]).reshape((1, -1))
    print(Y)

    nn = NeuralNetwork([2, 3, 2, 1])
    nn.run(X, Y, 0.01, 10000)

    test = np.array([0, 0]).reshape((2, 1))
    print(test.shape)
    print(nn.predict(test))
