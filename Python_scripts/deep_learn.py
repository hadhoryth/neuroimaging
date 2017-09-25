import tensorflow as tf
from tensorflow.python.framework import ops
from Logger import Log
import numpy as np
from Helper import Helpers
from data_learn import Analysis
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from features_learn import Features


class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers

    def _create_placeholders(self, n_x, n_y):
        x = tf.placeholder(tf.float32, shape=[n_x, None], name='X')
        y = tf.placeholder(tf.float32, shape=[n_y, None], name='Y')
        return x, y

    def _initialize_parameters(self):
        self.parameters = dict()
        for l in range(1, len(self.layers)):
            w, b = [self.layers[l], self.layers[l - 1]], [self.layers[l], 1]
            self.parameters['W' + str(l)] = tf.get_variable('W' + str(l), w,
                                                            initializer=tf.contrib.layers.xavier_initializer(seed=1))
            self.parameters['b' + str(l)] = tf.get_variable('b' +
                                                            str(l), b, initializer=tf.zeros_initializer())

    def _regularize(self):
        regularizers = tf.nn.l2_loss(self.parameters['W1'])
        for l in range(2, len(self.layers)):
            regularizers += tf.nn.l2_loss(self.parameters['W' + str(l)])
        return regularizers

    def _one_hot_matrix(self, labels, C):
        C = tf.constant(C, name='C')
        return tf.one_hot(labels, C, axis=0)

    def _forward_propagation(self, X, dropout=False):
        A = X
        L = len(self.layers)
        for l in range(1, L - 1):
            Z = tf.add(
                tf.matmul(self.parameters['W' + str(l)], A), self.parameters['b' + str(l)])
            A = tf.nn.relu(Z)
            if dropout:
                A = tf.nn.dropout(A, 0.2)
        ZL = tf.add(
            tf.matmul(self.parameters['W' + str(L - 1)], A), self.parameters['b' + str(L - 1)])
        return ZL

    def _random_mini_batches(self, X, Y, mini_batch_size=64, seed=0):
        m = X.shape[1]                  # number of training examples
        mini_batches = []
        np.random.seed(seed)

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        # number of mini batches of size mini_batch_size in your partitionning
        num_complete_minibatches = math.floor(m / mini_batch_size)
        for k in range(num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size: k *
                                      mini_batch_size + mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k *
                                      mini_batch_size + mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:,
                                      num_complete_minibatches * mini_batch_size: m]
            mini_batch_Y = shuffled_Y[:,
                                      num_complete_minibatches * mini_batch_size: m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def _compute_cost(self, ZL, true_labels):
        logits = tf.transpose(ZL)
        labels = tf.transpose(true_labels)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))
        return cost

    def model(self, X_train, Y_train, X_test, Y_test, X_valid, Y_valid, learning_rate=0.001,
              num_epochs=1500, minibatch_size=32, print_cost=True, reg_beta=None):

        ops.reset_default_graph()
        tf.set_random_seed(1)
        (n_x, m) = X_train.shape
        Y_train = self._one_hot_matrix(Y_train, 3)  # tensor
        Y_test = self._one_hot_matrix(Y_test, 3)
        Y_valid = self._one_hot_matrix(Y_valid, 3)
        n_y = Y_train.shape[0].value
        costs = []

        X, Y = self._create_placeholders(n_x, n_y)
        self._initialize_parameters()
        ZL = self._forward_propagation(X)
        cost = self._compute_cost(ZL, Y)

        if reg_beta is not None:
            cost = tf.reduce_mean(cost + reg_beta * self._regularize())

        # global_step = tf.Variable(0)
        # learning_rate = tf.train.exponential_decay(
        #     learning_rate, global_step, 100000, 0.96, staircase=True)

        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(cost)

        init = tf.global_variables_initializer()
        seed = 0
        with tf.Session() as session:
            session.run(init)
            Y_train, Y_test, Y_valid = session.run(
                Y_train), session.run(Y_test), session.run(Y_valid)

            for epoch in range(num_epochs):
                epoch_cost = 0.
                global_step = epoch_cost
                num_minibatches = int(m / minibatch_size)
                seed = seed + 1
                minibatches = self._random_mini_batches(
                    X_train, Y_train, minibatch_size, seed)
                for minibatch in minibatches:
                    (minibatch_X, minibatch_Y) = minibatch
                    _, minibatch_cost = session.run([optimizer, cost], feed_dict={
                        X: minibatch_X, Y: minibatch_Y})
                    epoch_cost += minibatch_cost / num_minibatches

                if print_cost == True and epoch % 100 == 0:
                    Log.info('Neural Network', 'Cost after epoch %i: %f' %
                             (epoch, epoch_cost))
                if print_cost == True and epoch % 5 == 0:
                    costs.append(epoch_cost)

            # plot the cost
            # plt.plot(np.squeeze(costs))
            # plt.ylabel('cost')
            # plt.xlabel('iterations (per tens)')
            # plt.title("Learning rate =" + str(learning_rate))

            # lets save the parameters in a variable
            parameters = session.run(self.parameters)
            ZL = self._forward_propagation(X)
            correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))

            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            Log.info(
                "Evaluation", 'Train - {0}'.format(accuracy.eval({X: X_train, Y: Y_train})))
            test_acc = accuracy.eval({X: X_test, Y: Y_test})
            Log.info("TEvaluation", 'Test - {0}'.format(test_acc))
            Log.info('Evaluation', 'Validation - %f' %
                     (accuracy.eval({X: X_valid, Y: Y_valid})))
            return test_acc, parameters
            # plt.show()

    def predict(self, X_test, Y_test, params):
        def covert_to_tensors(weigths):
            for l in range(1, len(self.layers)):
                self.parameters['W' +
                                str(l)] = tf.convert_to_tensor(weigths['W' + str(l)])
                self.parameters['b' +
                                str(l)] = tf.convert_to_tensor(weigths['b' + str(l)])

        covert_to_tensors(params)
        features = tf.placeholder(tf.float32, name='features')
        Y = tf.placeholder(tf.float32, name='Y')
        ZL = self._forward_propagation(features)
        p = tf.argmax(ZL)
        with tf.Session() as sess:
            prediction = sess.run(p, feed_dict={features: X_test})
            correct_prediction = tf.equal(prediction, tf.argmax(Y))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            acc = accuracy.eval({Y: Y_test})
        return acc


def get_test_sets(anls, data):
    train, test, labels_train, labels_test = anls.do_split(
        data['features'], data['labels'], test=0.5)
    dev = {'features': train, 'labels': labels_train}
    valid = {'features': test, 'labels': labels_test}

    Log.info('split', 'Normal - {0}; MCI - {1}; AD - {2}'.format(len(
        labels_train[labels_train == 0]), len(labels_train[labels_train == 1]), len(labels_train[labels_train == 2])))

    return dev, valid


if __name__ == '__main__':
    helper = Helpers()
    av45, fdg = helper.read_from_local(
        '/Users/XT/Documents/PhD/Granada/neuroimaging/Python_scripts/_cache/_all_data.pickle')
    keys = ['normal', 'mci', 'ad', 'labels_normal', 'labels_mci', 'labels_ad']

    av45 = av45['av45']
    anls = Analysis()
    mixed_av45 = anls.mix_data(av45, keys, test=0.1, logging=True)
    mixed_fdg = anls.mix_data(fdg['fdg'], keys, test=0, logging=True)

    dev_set, validation_set = get_test_sets(anls, mixed_av45['test'])

    # Mix FDG and AV45 data
    mixed_av45['train']['features'] = np.vstack(
        [mixed_av45['train']['features'], mixed_fdg['train']['features']])
    mixed_av45['train']['labels'] = np.append(
        mixed_av45['train']['labels'], mixed_fdg['train']['labels'])

    # ft = Features()
    # mixed_av45 = ft.resample_data(
    #     mixed_av45['train']['features'], mixed_av45['train']['labels'], r_type='tomek')

    # Training the model
    nn = NeuralNetwork([116, 116, 70, 60, 50, 20, 3])
    reg_betas = np.linspace(0.002, 0.003, 20)
    # reg_betas = [0.00278947368421]
    acc = []
    for i in range(len(reg_betas)):
        Log.info('Regularization param', reg_betas[i])
        a, params = nn.model(mixed_av45['train']['features'].T, mixed_av45['train']['labels'],
                             dev_set['features'].T, dev_set['labels'], validation_set['features'].T, validation_set['labels'], learning_rate=0.001, reg_beta=reg_betas[i])
        acc.append(a)
    # Log.info('Validation accuracy', nn.predict(
    #     validation_set['features'].T, validation_set['labels'], params))
