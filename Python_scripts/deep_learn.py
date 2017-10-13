import tensorflow as tf
from tensorflow.python.framework import ops
from Logger import Log
import numpy as np
from Helper import Helpers
from data_learn import Analysis
import math
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

    def _convert_to_hot_matrix(self, labels, C):
        C = tf.constant(C, name='C')
        with tf.Session() as sess:
            return sess.run(tf.one_hot(labels, C, axis=0))

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

    def model(self, X_train, Y_train, learning_rate=0.001,
              num_epochs=1500, minibatch_size=32, print_cost=True, reg_beta=None, use_cache=False):

        ops.reset_default_graph()
        tf.set_random_seed(1)
        self._initialize_parameters()
        cache_path = '_cache/model_params.pickle'
        if use_cache:
            h = Helpers()
            import os.path
            if os.path.isfile(cache_path):
                return h.read_from_local(cache_path)

        (n_x, m) = X_train.shape
        Y_train = self._convert_to_hot_matrix(Y_train, 3)
        n_y = Y_train.shape[0]
        costs = []
        X, Y = self._create_placeholders(n_x, n_y)

        ZL = self._forward_propagation(X)
        cost = self._compute_cost(ZL, Y)

        if reg_beta is not None:
            cost = tf.reduce_mean(cost + reg_beta * self._regularize())

        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(cost)

        init = tf.global_variables_initializer()
        seed = 0
        with tf.Session() as session:
            session.run(init)
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

                if print_cost and epoch % 100 == 0:
                    Log.info('Neural Network', 'Cost after epoch %i: %f' %
                             (epoch, epoch_cost))
                if print_cost and epoch % 5 == 0:
                    costs.append(epoch_cost)
            model_parameters = session.run(self.parameters)
            if use_cache:
                h.dump_to_local(cache_path, model_parameters)
            return model_parameters

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
            acc = accuracy.eval({Y: self._convert_to_hot_matrix(Y_test, 3)})
            cm = sess.run(tf.confusion_matrix(
                np.asarray(Y_test, dtype='int'), prediction, 3))
            f_score = self._get_precision_reall(np.asarray(
                Y_test, dtype='int'), prediction)
            return acc, cm, f_score

    def _get_precision_reall(self, true_labels, predicted):
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(
            predicted, true_labels, average='macro')
        recall = recall_score(
            predicted, true_labels, average='macro')

        f_score = f1_score(true_labels, predicted, average='macro')

        Log.info('Validation', 'Precision {0}'.format(precision))
        Log.info('Validation', 'Recall {0}'.format(recall))
        Log.info('Validation', 'F1 score {0}'.format(f_score))
        return f_score


def get_test_sets(anls, data):
    train, test, labels_train, labels_test = anls.do_split(
        data['features'], data['labels'], test=0.5)
    dev = {'features': train, 'labels': labels_train}
    valid = {'features': test, 'labels': labels_test}

    Log.info('split', 'Normal - {0}; MCI - {1}; AD - {2}'.format(len(
        labels_train[labels_train == 0]), len(labels_train[labels_train == 1]), len(labels_train[labels_train == 2])))

    return dev, valid


if __name__ == '__main__':
    helper, ft = Helpers(), Features()
    av45, fdg = helper.read_from_local(
        '/Users/XT/Documents/PhD/Granada/neuroimaging/Python_scripts/_cache/_all_data.pickle')
    keys = ['normal', 'mci', 'ad', 'labels_normal', 'labels_mci', 'labels_ad']

    av45 = av45['av45']
    anls = Analysis()
    mixed_av45 = anls.mix_data(av45, keys, test=0.1, logging=True)
    mixed_fdg = anls.mix_data(fdg['fdg'], keys, test=0.1, logging=True)

    gl_mixing = Helpers.concat_dicts(mixed_av45['test'], mixed_fdg['test'])

    # ft = Features()
    # mixed_av45['test'] = ft.resample_data(
    #     mixed_av45['test']['features'], mixed_av45['test']['labels'])
    # mixed_av45['test'] = {'features': mixed_av45['test']
    #                       [0], 'labels': mixed_av45['test'][1]}

    dev_set, validation_set = get_test_sets(anls, mixed_av45['test'])

    # Mix FDG and AV45 data
    mixed_av45['train']['features'] = np.vstack(
        [mixed_av45['train']['features'], mixed_fdg['train']['features']])
    mixed_av45['train']['labels'] = np.append(
        mixed_av45['train']['labels'], mixed_fdg['train']['labels'])

    duplicates_dev = ft.check_for_duplicates(
        mixed_av45['train']['features'], dev_set['features'], 'Dev: ')
    duplicates_valid = ft.check_for_duplicates(
        mixed_av45['train']['features'], validation_set['features'], 'Validation: ')

    # Training the model
    nn = NeuralNetwork([116, 116, 70, 60, 50, 20, 3])
    # reg_betas = np.linspace(0.002, 0.003, 20)
    reg_betas = [0.00278947368421]
    acc = []
    for i in range(len(reg_betas)):
        Log.info('Regularization param', reg_betas[i])
        weigths = nn.model(mixed_av45['train']['features'].T, mixed_av45['train']['labels'],
                           learning_rate=0.001, reg_beta=reg_betas[i], use_cache=True)

    dev_acc, dev_cm, _ = nn.predict(
        dev_set['features'].T, dev_set['labels'], weigths)
    Log.info('Validation', 'Test {0}'.format(dev_acc))
    Log.info('Validation', 'Confusion matrix (dev) \n{0}'.format(dev_cm))

    valid_acc, valid_cm, _ = nn.predict(
        validation_set['features'].T, validation_set['labels'], weigths)
    Log.info('Validation', 'Validation {0}'.format(valid_acc))
    Log.info('Validation', 'Confusion matrix (valid) \n{0}'.format(valid_cm))
