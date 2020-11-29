import numpy as np


class Network:
    def __init__(self, *nodes, m=10, lr=2):
        self.a = []
        self.mini = m
        self.layers = len(nodes)
        self.sizes = nodes
        self.weights = [np.random.randn(nodes[i - 1], nodes[i]) for i in range(1, len(nodes))]
        self.biases = [np.zeros([1, nodes[i]]) for i in range(1, len(nodes))]
        self.lr = lr

    def save_weights_biases(self):
        np.savez("weights", *self.weights)
        np.savez("biases", *self.biases)

    def load_weights_biases(self):
        w = np.load("weights.npz")
        b = np.load("biases.npz")

        self.weights = [w[i] for i in w.files]
        self.biases = [b[i] for i in b.files]

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def sigmoid_der(self, s):
        return s * (1 - s)

    def cross_softmax_der(self, output, expected):
        return output - expected

    def softmax(self, s):
        exps = np.exp(s - np.max(s, axis=1, keepdims=True))
        return exps/np.sum(exps, axis=1, keepdims=True)

    def relu(self, s):
        return s * (s > 0)

    def relu_der(self, s):
        return 1 * (s > 0)



    def feed_forward(self, ip):
        cv = ip
        self.a = []

        self.a.append(cv)
        for i in range(len(self.weights)-1):
            z = np.dot(cv, self.weights[i]) + self.biases[i]
            cv = self.sigmoid(z)
            # cv = self.relu(z)
            self.a.append(cv)

        z = np.dot(cv, self.weights[-1]) + self.biases[-1]
        cv = self.softmax(z)
        self.a.append(cv)

    def backprop(self, actual):

        a_der = self.cross_softmax_der(self.a[-1], actual) # technically a_der * z_der
        a_der /= self.mini # take average by dividing first, because everything else multiplies with this

        # this is for second to last set of weights,
        # has to be before self.weights is changed
        w_der = np.dot(a_der, self.weights[-1].T)

        self.weights[-1] -= self.lr * np.dot(self.a[-2].T, a_der)
        self.biases[-1] -= self.lr * np.sum(a_der, axis=0, keepdims=True)

        for layer in range(self.layers-2, 0, -1):

            a_der = w_der * self.sigmoid_der(self.a[layer])
            # a_der = w_der * self.relu_der(self.a[layer])
            w_der = np.dot(a_der, self.weights[layer-1].T)

            self.weights[layer-1] -= self.lr * np.dot(self.a[layer-1].T, a_der)
            self.biases[layer-1] -= self.lr * np.sum(a_der, axis=0)

    def train(self, ip, res):
        for i in range(len(ip) // self.mini):
            self.feed_forward(ip[self.mini*i:self.mini*(i+1)])
            self.backprop(res[self.mini*i:self.mini*(i+1)])

    def test(self, ip, res):
        self.feed_forward(ip)

        output = self.a[-1].argmax(axis=1)
        expected = np.argmax(res, axis=1)
        print(output)
        # print(expected)
        return np.mean(output==expected)