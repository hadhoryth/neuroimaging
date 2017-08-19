import os.path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier

from colorama import init, Fore
init(autoreset=True)


class Analysis:

    def getClassifier(self, type, k):
        if type.lower() == 'knn':
            print('Nearest Neighbor Classifier has been selected')
            clf = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')
        elif type.lower() == 'svm':
            print('Support vector machine has been selected')
            tuned_param = {'C': [0.1, 10, 10, 100], 'gamma': [0.001, 0.01, 0.1, 0.2]}
            clf = GridSearchCV(SVC(kernel='rbf', decision_function_shape='ovr', random_state=0, class_weight='balanced'), tuned_param, cv=10)
        elif type.lower() == 'lin_svm':
            print('Linear Support Vector machine has been selected')
            clf = OneVsRestClassifier(LinearSVC(random_state=0, C=10))
        return clf

    def getTrainingSet(self, data, size, random, keys):
        step = len(keys) // 2
        features, labels = data[keys[0]], data[keys[step]]
        for key in range(1, step):
            features = np.concatenate((features, data[keys[key]]))
            labels = np.concatenate((labels, data[keys[key + step]]))

        return train_test_split(features, labels, test_size=size, random_state=random)

    def print_data_split(self, train_size, test_size):
        print('Train fraction: ', end="")
        print(Fore.GREEN + '{0}'.format(train_size), end="")
        print('\nTest fraction: ', end="")
        normal, mci, ad = test_size[test_size == 0], test_size[test_size == 1], test_size[test_size == 2]
        print(Fore.GREEN + str(len(test_size)) + ' where: Normal: ' + str(len(normal)) + '; MCI: ' + str(len(mci)) + '; AD: ' + str(len(ad)))
        print('Total data points: ', end="")
        print(Fore.GREEN + '{0}'.format(train_size + len(test_size)))

    def performPCA(self, data, n_pca):
        pca = PCA(n_components=n_pca).fit(data)
        return pca.transform(data)

    def readOrTrainClassifier(self, clf, name, features_train, labels_train, force=False):
        cached_clf = os.path.join('_cache', name + '_classifier' + '.pickle')
        if os.path.isfile(cached_clf) and not force:
            print('Loading classifier from cached files ........ ', end='')
            load = joblib.load(cached_clf)
            print('OK')
            return load
        else:
            print('Saving classifier to cache ........ ', end='')
            clf.fit(features_train, labels_train)
            joblib.dump(clf, cached_clf, compress=9)
            print('OK')
            return clf

    def plotMulticlassROC(self, n_classes, labels_test, pred_labels):
        fpr, tpr, roc_auc = dict(), dict(), dict()
        import pdb
        pdb.set_trace()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(labels_test, pred_labels[:, i])

    def classify(self, data, keys, classifier, k=2, training_split=[0.25, 42], apply_pca=False, n_pca=10, printing=True, scaling=False, useCache=True, clf_cache_name='_clf'):
        self.keys = keys

        features_train, labels_train = data['train'], data['labels_train']
        features_test, labels_test = data['test'], data['labels_test']

        if printing:
            self.print_data_split(len(features_train), labels_test)

        if scaling:
            scaler = StandardScaler().fit(features_train)
            features_train = scaler.transform(features_train)

        if apply_pca:
            features_train = self.performPCA(features_train, n_pca)

        clf = self.readOrTrainClassifier(self.getClassifier(classifier, k), clf_cache_name, features_train, labels_train, force=True)

        pred = clf.predict(features_test)

        # if printing:
        #     self.plotMulticlassROC(3, labels_test, clf.decision_function(features_test))

        return clf.best_score_, confusion_matrix(labels_test, pred)

    def plotDataWithPCA(self, data, mode='3D'):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        import seaborn as sns

        n_components = 3
        if mode == '2D':
            n_components = 2

        data_pca = self.performPCA(data, n_components)
        print(data)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection=mode.lower())
        ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2])
        plt.show()