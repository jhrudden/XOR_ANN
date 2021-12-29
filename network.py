import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 1
ROUNDS = 1000


class NeuralNet:
    def __init__(self):
        self.dimensions = [2, 2, 1]
        self.weights = {}
        self.biases = {}
        np.random.seed(2)
        for l in range(1, len(self.dimensions)):
            layer_neurons = self.dimensions[l]
            prev_layer_neurons = self.dimensions[l - 1]
            W = np.random.normal(
                0, 1, (layer_neurons, prev_layer_neurons))
            B = np.random.random((layer_neurons))
            self.weights[l] = W
            self.biases[l] = B

    def sigmoid(self, linear_hypothesis):
        """Z vector of weight based on connection recieved, neuron input, and bias"""
        sig = 1 / (1 + np.exp(-linear_hypothesis))
        return sig

    def sigmoid_derivative(self, linear_hypothesis):
        d_s = np.multiply(self.sigmoid(linear_hypothesis),
                          np.subtract(1, self.sigmoid(linear_hypothesis)))
        return d_s

    def forward_propagation(self, X):
        cache = {}
        for l in range(1, len(self.dimensions)):
            weights = self.weights[l]
            biases = self.biases[l]
            w_dot_x = [np.dot(weight, X) for weight in weights]
            weighted_averages = np.add(w_dot_x, biases)
            activations = [self.sigmoid(x) for x in weighted_averages]
            layers_cache = [(X, weights, weighted_averages)]
            X = activations
            cache[l] = layers_cache
        return X[0], cache

    def cost(self, prediction, known):
        return np.divide((prediction - known) ** 2, 2)

    def d_cost(self, prediction, known):
        return prediction - known

    def back_propagation(self, known, prediction, cache):
        bias_errors = []
        weight_errors = []
        # errors of Final Layer
        L_inputs, weights, L_weighted_inputs = cache[2][0]
        L_error = np.multiply(self.d_cost(prediction, known),
                              self.sigmoid_derivative(L_weighted_inputs))
        bias_errors.append(L_error)

        weight_errors.append(np.multiply(L_inputs, L_error))

        # errors of hidden layer
        l_inputs, _, l_weighted_inputs = cache[1][0]
        l_cost_gradient = np.dot(np.transpose(weights), L_error)
        l_error = np.multiply(
            l_cost_gradient, self.sigmoid_derivative(l_weighted_inputs))

        bias_errors.insert(0, l_error)
        l_weight_errors = np.multiply(l_inputs, l_error)
        weight_errors.insert(0, l_weight_errors)

        return weight_errors, bias_errors

    def update_params(self, w_errors, b_errors):
        new_w = {}
        new_b = {}
        for l in range(1, len(self.dimensions)):
            l_weights = np.subtract(
                self.weights[l], np.multiply(w_errors[l - 1], LEARNING_RATE))
            l_biases = np.subtract(
                self.biases[l], np.multiply(b_errors[l - 1], LEARNING_RATE))
            new_w[l] = l_weights
            new_b[l] = l_biases
        self.weights = new_w
        self.biases = new_b

    def train(self, training_data, epochs):
        costs = []
        t_size = len(training_data)
        for i in range(epochs):
            sample = np.random.choice(list(range(t_size)), t_size)
            weight_error = 0
            bias_error = 0
            cost = 0
            for index in sample:
                data = training_data[index]
                X, Y_hat = data
                Y, cache = self.forward_propagation(X)
                C = self.cost(Y, Y_hat)
                cost += C
                w_e, b_e = self.back_propagation(Y_hat, Y, cache)
                weight_error = np.add(weight_error, w_e)
                bias_error = np.add(bias_error, b_e)
            costs.append(cost / 4)
            self.update_params(np.divide(weight_error, 4),
                               np.divide(bias_error, 4))
        return costs

    def predict(self, X):
        Y, _ = self.forward_propagation(X)
        return Y


data = [
    ([1, 0], 1),
    ([1, 1], 0),
    ([0, 1], 1),
    ([0, 0], 0),
]


net = NeuralNet()
cost = net.train(data, ROUNDS)
plt.plot(range(ROUNDS), cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()
