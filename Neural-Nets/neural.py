import theano
import theano.tensor as T
from theano.tensor.nnet import softmax, sigmoid, binary_crossentropy
import numpy as np


class NeuralNetwork(object):
    def __init__(self, layers):
        self.layers = layers

        self.x = T.matrix('x')
        self.y = T.matrix('y')

        layers[0].set_inpt(self.x)
        for i in range(1, len(layers)):
            prev_layer, curr_layer = layers[i - 1], layers[i]
            curr_layer.set_inpt(prev_layer.output)

        self.params = [param for layer in layers for param in layer.params]

        self.tf_feedforward = theano.function(inputs=[self.x], outputs=layers[-1].a)

    def feedforward(self, x):
        return self.tf_feedforward(x)

    def SGD(self, training_data, epochs, eta, mini_batch_size, test_data=None):
        """
        SGD stands for Stochastic gradient descent
        """

        output_layer = self.layers[-1]

        cost = output_layer.cost(self.y)
        grads = T.grad(cost, self.params)
        updates = [(param, param - eta * grad) for param, grad in zip(self.params, grads)]

        # uncomment to visualize the computational graph
        # theano.printing.pydotprint(cost, outfile='cost.png', var_with_name_simple=True)

        i = T.lscalar('i')  # mini batch index

        tf_train_mb = theano.function(
            inputs=[i], outputs=cost,
            givens={
                self.x:
                    training_data[0][i * mini_batch_size:(i + 1) * mini_batch_size],
                self.y:
                    training_data[1][i * mini_batch_size:(i + 1) * mini_batch_size]},
            updates=updates)

        tf_test_accuracy = theano.function(
            inputs=[i], outputs=self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                    test_data[0][i * mini_batch_size:(i + 1) * mini_batch_size],
                self.y:
                    test_data[1][i * mini_batch_size:(i + 1) * mini_batch_size]},
            updates=updates)

        n_training_batches = training_data[0].get_value().shape[0] // mini_batch_size
        if test_data:
            n_test_batches = test_data[0].get_value().shape[0] // mini_batch_size

        for epoch in range(epochs):
            print('Current epoch: {}'.format(epoch))

            if test_data:
                accuracy = np.mean([tf_test_accuracy(mbi) for mbi in range(n_test_batches)])
                print('Accuracy on test data: {0:.4f}\n'.format(accuracy))

            for mbi in range(n_training_batches):
                curr_cost = tf_train_mb(mbi)

    def __str__(self):
        return 'NeuralNetwork(layers={})'.format(self.layers)


class FullyConnectedLayer(object):
    """A fully connected hidden layer"""

    def __init__(self, n_in, n_out, activation_fn=sigmoid):
        """`n_in` is the number of inputs
        `n_out` is the number of outputs"""
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn

        self.w = theano.shared(
            np.asarray(
                np.random.normal(size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='weight',
            borrow=True)

        self.b = theano.shared(
            np.asarray(
                np.random.normal(size=(n_out,)),
                dtype=theano.config.floatX),
            name='bias',
            borrow=True)

        self.params = (self.w, self.b)

    def set_inpt(self, inpt):
        inpt = inpt.reshape((-1, self.n_in))
        self.output = self.activation_fn(inpt.dot(self.w) + self.b)

    def __str__(self):
        return 'FullyConnectedLayer({0}, {1}, {2})'.format(
            self.n_in, self.n_out, self.activation_fn.__name__)

    def __repr__(self):
        return str(self)


class QuadraticCostLayer(object):
    def __init__(self, n_in):
        self.n_in = n_in
        self.params = ()

    def set_inpt(self, inpt):
        inpt = inpt.reshape((-1, self.n_in))
        self.a = inpt

    def cost(self, y):
        return T.mean((y - self.a) ** 2) / 2

    def accuracy(self, y):
        return T.mean(T.eq(T.argmax(self.a, axis=1), T.argmax(y, axis=1)))

    def __str__(self):
        return 'QuadraticCostLayer({0})'.format(self.n_in)

    def __repr__(self):
        return str(self)


class CrossEntropyCostLayer(object):
    def __init__(self, n_in):
        self.n_in = n_in
        self.params = ()

    def set_inpt(self, inpt):
        inpt = inpt.reshape((-1, self.n_in))
        self.a = inpt

    def cost(self, y):
        return T.mean(binary_crossentropy(self.a, y))

    def accuracy(self, y):
        return T.mean(T.eq(T.argmax(self.a, axis=1), T.argmax(y, axis=1)))

    def __str__(self):
        return 'CrossEntropyCostLayer({0})'.format(self.n_in)

    def __repr__(self):
        return str(self)
