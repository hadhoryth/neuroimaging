import tensorflow as tf
from neat.graphs import feed_forward_layers
from neat.six_util import itervalues
from itertools import count
import numpy as np
from tensorflow.python.framework import ops


class _Activations:
    def _sigmoid(z):
        return tf.nn.sigmoid(z)

    def _relu(z):
        return tf.nn.relu(z)

    def _tanh(z):
        return tf.nn.tanh(z)

    activations = dict(sigmoid=_sigmoid, relu=_relu, tanh=_tanh)


class Node:
    def __init__(self, activation, prev_size, idx, weights, bias):
        self.activ_func = _Activations.activations.get(activation, 'Unknown')
        self.id = idx
        self._init_weights(prev_size, weights, bias)

    def _init_weights(self, prev_size, weights, bias):
        if (weights is None or len(weights) == 0):
            self.weights = tf.get_variable('W' + str(self.id), shape=(
                prev_size, 1), initializer=tf.contrib.layers.xavier_initializer(seed=1))
            self.bias = tf.get_variable('b' + str(self.id), shape=(1, 1),
                                        initializer=tf.zeros_initializer())
        else:
            self.weights = tf.Variable(tf.convert_to_tensor(weights),
                                       name='W' + str(self.id))
            self.bias = tf.Variable(tf.convert_to_tensor(bias),
                                    name='b' + str(self.id))


class Layer:
    def __init__(self, **kwargs):
        if not kwargs.get('genes'):
            self.num_nodes = kwargs['prev_size']
            return
        self.num_nodes = len(kwargs['genes'])
        self.nodes = dict()
        for gene in kwargs['genes']:
            self.nodes[gene.key] = Node(
                gene.activation, kwargs['prev_size'], gene.key, gene.weights, gene.bias)

    def feed_forward(self, ftr, activate):
        out = list()
        for node in self.nodes.values():
            z = tf.add(tf.matmul(tf.transpose(node.weights), ftr), node.bias)
            if activate:
                z = node.activ_func(z)
            out.append(z)
        return tf.concat(out, axis=0)


class TensorFeedForward(object):
    def __init__(self, genome, config):
        ops.reset_default_graph()
        tf.set_random_seed(54)
        in_nodes = len(config.genome_config.input_keys)
        self.layers = list([Layer(prev_size=in_nodes)])
        self.elements = in_nodes
        self.item_generator = count(in_nodes)
        self._create(genome, config)
        self.num_layers = len(self.layers)

    def _add_layer(self, funcs, keys, tp='in'):
        prev_size = self.layers[-1].num_nodes
        self.layers.append(Layer(len(funcs), keys, prev_size=prev_size,
                                 funcs=funcs, cnt=self.elements))
        self.elements += len(funcs)

    def _create(self, genome, config):
        connections = [cg.key for cg in itervalues(
            genome.connections) if cg.enabled]
        struct = feed_forward_layers(config.genome_config.input_keys,
                                     config.genome_config.output_keys,
                                     connections)
        for layer in struct:
            prev_size = self.layers[-1].num_nodes
            nn_layer = Layer(genes=[genome.nodes[key] for key in layer],
                             prev_size=prev_size)
            self.layers.append(nn_layer)

    def _create_placeholders(self, n_x, n_y, tp=tf.float32):
        x = tf.placeholder(tp, shape=[n_x, None], name='X')
        y = tf.placeholder(tp, shape=[n_y, None], name='Y')
        return x, y

    def forward_propagate(self, A):
        activate = True
        for layer in self.layers[1:]:
            if layer is self.layers[-1] and self.layers[-1].num_nodes > 1:
                activate = False
            A = layer.feed_forward(A, activate)
        return A

    def _get_minibatches(self, epoch, X, Y, batch_size=32):
        m, n = X.shape[1], X.shape[1] // batch_size
        mini_batches = list()
        np.random.seed(epoch + 1)

        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

        for k in range(n):
            mini_batch_X = shuffled_X[:, k * batch_size: k *
                                      batch_size + batch_size]
            mini_batch_Y = shuffled_Y[:, k * batch_size: k *
                                      batch_size + batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % batch_size != 0:
            mini_batch_X = shuffled_X[:, n * batch_size: m]
            mini_batch_Y = shuffled_Y[:, n * batch_size: m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def optimize(self, features, labels, num_epoch=150, learn=0.01):
        """ Optimizes the neural net weights with back propagation algorithm
            Parameters
            ----------
            features: size of (num_of_features, num_data_points)
            labels: size of (num_data_points, 1)
            Return
            ---------
            `Accuracy` float          
        """
        self.session = tf.Session()
        if self.num_layers < 2:
            return -1
        depth = self.layers[-1].num_nodes
        hot_labels = self.session.run(
            tf.transpose(tf.one_hot(np.ravel(labels), depth)))
        X, Y = self._create_placeholders(features.shape[0], depth)
        logits = tf.transpose(self.forward_propagate(X))

        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.transpose(Y)))

        optimizer = tf.train.AdamOptimizer(
            learning_rate=learn).minimize(cost)

        self.session.run(tf.global_variables_initializer())
        for i in range(num_epoch):
            epoch_cost = 0.0
            minibatches = self._get_minibatches(i, features, hot_labels)
            for batch in minibatches:
                try:
                    _, minibatch_cost = self.session.run(
                        [optimizer, cost], feed_dict={X: batch[0], Y: batch[1]})
                except ValueError:
                    print('After iterating')
                epoch_cost += minibatch_cost / len(minibatches)
        return self.predict(features, labels, False)

    def predict(self, features, labels, close_session=True):
        X, _ = self._create_placeholders(features.shape[0], None)
        hot = tf.one_hot(np.ravel(labels), np.max(labels) + 1)
        p1 = tf.argmax(tf.nn.softmax(logits=self.forward_propagate(X)), axis=0)

        prediction, hot_Y = self.session.run([p1, hot], {X: features})
        correct_prediction = tf.equal(prediction, np.ravel(labels))
        accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, "float")).eval(session=self.session)

        if close_session:
            self.session.close()

        return float('%.3f' % accuracy)

    def _get_f_score(self, predicted, Y_true):

        TP = tf.count_nonzero(predicted * Y_true)
        TN = tf.count_nonzero((predicted - 1) * (Y_true - 1))
        FP = tf.count_nonzero(predicted * (Y_true - 1))
        FN = tf.count_nonzero((predicted - 1) * Y_true)
        precision = tf.divide(TP, TP + FP)
        recall = tf.divide(TP, TP + FN)
        f1 = 2 * tf.divide(tf.multiply(precision, recall),
                           tf.add(precision, recall)).eval(session=self.session)
        if np.isnan(f1):
            f1 = 0.1
        return f1

    def update_genome(self, genome):
        """ Saving backpropagated **_weights_** and **bias** to the original genome """
        network_nodes = dict()
        for layer in self.layers[1:]:
            network_nodes.update(layer.nodes.items())

        for key in genome.nodes.keys():
            if network_nodes.get(key) is None:
                continue

            bias = network_nodes[key].bias.eval(session=self.session)
            genome.nodes[key].weights = network_nodes[key].weights.eval(
                session=self.session)
            genome.nodes[key].bias = np.ravel(bias)[0]
            del bias

        self.session.close()
        return genome
