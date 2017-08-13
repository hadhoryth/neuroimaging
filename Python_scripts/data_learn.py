import os.path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.externals import joblib

from colorama import init, Fore
init(autoreset=True)


class Analysis:

    def getClassifier(self, type, k, data_train, label_train):
        if type.lower() == 'knn':
            print('Nearest Neighbor Classifier has been selected')
            clf = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')
        elif type.lower() == 'svm':
            print('Support vector machine has been selected')
            tuned_param = {'C': [0.1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 0.2], 'kernel': ['rbf'], 'class_weight': ['balanced']}
            clf = GridSearchCV(SVC(), tuned_param, cv=10)
        clf.fit(data_train, label_train)
        return clf

    def getTrainingSet(self, data, size, random):
        step = len(self.keys) // 2
        features, labels = data[self.keys[0]], data[self.keys[step]]
        for key in range(1, step):
            features = np.concatenate((features, data[self.keys[key]]))
            labels = np.concatenate((labels, data[self.keys[key + step]]))

        return train_test_split(features, labels, test_size=size, random_state=random)

    def print_data_split(self, train_size, test_size):
        print('----------------------------------------------------')
        print('Train fraction: ', end="")
        print(Fore.GREEN + '{0}'.format(train_size), end="")
        print('\nTest fraction: ', end="")
        print(Fore.GREEN + '{0}'.format(test_size))
        print('Total data points: ', end="")
        print(Fore.GREEN + '{0}'.format(train_size + test_size))
        print('----------------------------------------------------')

    def performPCA(self, data, n_pca):
        pca = PCA(n_components=n_pca).fit(data)
        return pca.transform(data)

    def readOrTrainClassifier(self, classifier, k, features_train, labels_train, useCache, name):
        cached_clf = name + '_classifier.pickle'
        if useCache and os.path.isfile(cached_clf):
            return joblib.load(cached_clf)
        else:
            clf = self.getClassifier(classifier, k, features_train, labels_train)
            save_clf = joblib.dump(clf.best_estimator_, cached_clf, compress=9)
            return clf

    def classify(self, data, keys, classifier, k=2, training_split=[0.25, 42], apply_pca=False, n_pca=10, printing=True, scaling=False, useCache=True, clf_cache_name='clf'):
        self.keys = keys

        features_train, features_test, labels_train, labels_test = self.getTrainingSet(data, training_split[0], training_split[1])
        if printing:
            self.print_data_split(len(features_train), len(features_test))

        if scaling:
            scaler = StandardScaler().fit(features_train)
            features_train = scaler.transform(features_train)

        if apply_pca:
            features_train = self.performPCA(features_train, n_pca)

        # clf = self.readOrTrainClassifier(classifier, k, features_train, labels_train, useCache, clf_cache_name)
        # scores = cross_val_score(, features_train, labels_train, cv=10, scoring='accuracy')

        clf = self.getClassifier(classifier, k, features_train, labels_train)
        return clf.best_score_
        # return clf.score, [[labels_test], [clf.predict(features_test)]]
