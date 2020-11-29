import numpy as np


class Network:
    def __init__(self, *nodes, lr=1):
        self.mini = 1000
        self.layers = len(nodes)
        self.reset()
        self.weights = []
        self.biases = []
        self.lr = lr

        for i in range(1, len(nodes)):
            self.weights.append(np.random.randn(nodes[i-1], nodes[i]))
            self.biases.append(np.zeros(nodes[i]))

#     def cost_der(self, output, expected):
#         return [2 * (output[i] - expected[i]) for i in range(self.mini)]

    def feed_forward(self, input):

        a_values = [input]
        z_values = [np.array([0])]

        for i in range(len(self.weights)):
            z_values.append(np.dot(a_values[-1], self.weights[i]) + self.biases[i])
            a_values.append(sigmoid(z_values[-1]))


        self.a.append(a_values)
        self.z.append(z_values)

    def error(self, pred, real):
        percentage = 0
        for i in range(len(pred)):
            print(pred[i], real[i])
            if np.argmax(pred[i]) == np.argmax(real[i]):
                percentage += 1

        return percentage / len(pred)

    def reset(self):
        self.a = []
        self.z = []

    def backprop(self, actual):

        weights_change = [np.zeros_like(i) for i in self.weights]
        biases_change = [np.zeros_like(i) for i in self.biases]

        a_der = self.cost_der([self.a[i][-1] for i in range(self.mini)], actual)
        sig_der = sigmoid_der(np.array([i[-1] for i in self.a]))

        for batch in range(self.mini):

            weights_change[-1] -= np.dot(self.a[batch][-2].reshape(-1, 1), np.reshape(a_der[batch] * sig_der[batch], (1, -1)))
            biases_change[-1] -= a_der[batch] * sig_der[batch]

        for layer in range(self.layers-2, 0, -1):

            a_der = [np.dot(np.reshape(a_der[batch] * sig_der[batch], (1, -1)), self.weights[layer].T) for batch in range(self.mini)]
            sig_der = [np.reshape(sigmoid_der(np.array(i[layer])), (1, -1)) for i in self.a]

            for batch in range(self.mini):

                weights_change[layer-1] -= np.dot(self.a[batch][layer-1].reshape(-1, 1), a_der[batch] * sig_der[batch])
                biases_change[layer-1] -= np.reshape(a_der[batch] * sig_der[batch], (-1))

        weights_change = [(weights_change[layer]/self.mini)*self.lr for layer in range(self.layers-1)]
        biases_change = [(biases_change[layer]/self.mini)*self.lr for layer in range(self.layers-1)]
        # print(weights_change)
        # print(biases_change)
        self.weights = [self.weights[layer] + weights_change[layer] for layer in range(self.layers-1)]
        self.biases = [self.biases[layer] + biases_change[layer] for layer in range(self.layers-1)]

    def train(self, ip, res):

        self.reset()
        for i in range(len(ip)):
            if i % 1000 == 0:
                print(i)
            if i > 0 and i % self.mini == 0:

                self.backprop(res[i-self.mini:i])
                self.reset()

            self.feed_forward(ip[i])

    def test(self, ip, res):
        percentage = 0
        self.reset()
        for i in range(len(ip)):
            self.feed_forward(ip[i])
            if np.argmax(self.a[0][-1]) == res[i]:
                percentage += 1
            self.reset()

        percentage /= len(res)
        return percentage


class NumberNetwork(Network):
    def __init__(self):
        super().__init__(784, 20, 20, 10)

    # def cost(self, output, expected):
    #     return sum((output-expected)**2)

    def cost_der(self, output, expected):
        # print(output[0].shape, expected[0].shape)
        return [2 * (output[i] - expected[i]) for i in range(self.mini)]

    def train_with_dataset(self):
        ip = [sample.flatten() / 255 for sample in np.load('mnist/train-images.npy')]
        actual_raw = np.load('mnist/train-labels.npy')
        # print(actual_raw)
        actual = np.zeros([len(actual_raw), 10])
        # print(actual[0:5])
        for i, n in enumerate(actual_raw):
            actual[i][n] = 1

        self.train(ip, actual)

    def test_with_dataset(self):
        ip = [sample.flatten() / 255 for sample in np.load('mnist/test-images.npy')]
        actual = np.load('mnist/test-labels.npy')
        return self.test(ip, actual)