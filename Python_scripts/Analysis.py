import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from colorama import init, Fore


init(autoreset=True)


class Classifier:

    def __init__(self, print=False):
        self.printing = print

    def reset(self):
        self.data, self.keys = [], []
        self.features_train, self.features_test, self.labels_train, self.labels_test = [], [], [], []

    def generateTrainTestSet(self):
        features = np.concatenate([self.data['normal'], self.data['ad']])
        labels = np.concatenate([self.data['labels_normal'], self.data['labels_ad']])
        self.features_train, self.features_test, self.labels_train, self.labels_test = train_test_split(features, labels, test_size=0.25, random_state=42)
        print('Train fraction: ', end="")
        print(Fore.GREEN + '{0}'.format(len(self.features_train)), end="")
        print('\nTest fraction: ', end="")
        print(Fore.GREEN + '{0}'.format(len(self.features_test)))
        print('Total data points: ', end="")
        print(Fore.GREEN + '{0}'.format(len(features)))

    def pickClassifier(self):
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10],
                      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
                      }
        clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced',
                               decision_function_shape='ovr'), param_grid)

        return clf

    def printClassifierResults(self, pred, target_names):
        print(classification_report(self.labels_test, pred,
                                    target_names=target_names))
        cm = confusion_matrix(self.labels_test, pred)
        print('TN: {0}, FP: {1}\nFN: {2}, TP: {3}'.format(cm[0, 0], cm[0, 1],
                                                          cm[1, 0], cm[1, 1]))

    def performPCA(self, n):
        pca = PCA(n_components=n).fit(self.features_train)
        features_train_pca, features_test_pca = [], []
        features_train_pca, features_test_pca = pca.transform(self.features_train), pca.transform(self.features_test)
        return features_train_pca, features_test_pca

    def optimizePCA(self, n, clf):
        features_train_pca, features_test_pca = self.performPCA(n)
        prediction, score = self.performClassification(clf, features_train_pca, self.labels_train, features_test_pca, self.labels_test)
        return prediction, score

    def performTSNE(self):
        standard_scalar = StandardScaler()
        features_train_tsne = standard_scalar.fit_transform(self.features_train)
        features_test_tsne = standard_scalar.fit_transform(self.features_test)
        tsne = TSNE(n_components=2, random_state=0)
        features_train_tsne_2d = tsne.fit_transform(features_train_tsne)
        features_test_tsne_2d = tsne.fit_transform(features_test_tsne)

        # if self.printing:
        # markers = ['o', '*']
        # color_map = {0: 'red', 1: 'blue'}
        # plt.figure()
        # for idx, cl in enumerate(np.unique(self.labels_train)):
        #     plt.scatter(x=features_train_tsne_2d[self.labels_train == cl, 0],
        #                 y=features_train_tsne_2d[self.labels_train == cl, 1],
        #                 c=color_map[idx], marker=markers[idx], label=cl)
        # # plt.scatter(features_train_tsne_2d[:, 0], features_train_tsne_2d[:, 1],
        # #             c='r', cmap=plt.cm.Spectral)
        # plt.xlabel('X in t-SNE')
        # plt.ylabel('Y in t-SNE')
        # plt.legend(loc='upper left')
        # plt.title('t-SNE visualization')
        # plt.show()

        return features_train_tsne_2d, features_test_tsne_2d

    def performClassification(self, clf, x_train, y_train, x_test, y_test):
        clf.fit(x_train, y_train)
        prediction = clf.predict(x_test)
        score = accuracy_score(y_test, prediction)
        return prediction, score
        # return accuracy_score(self.features_train, prediction)

    def classify(self, in_data, data_keys, applyPCA=False, pca_comp=5, applyTSNE=False):
        self.reset()
        self.data, self.keys = in_data, data_keys

        self.generateTrainTestSet()
        self.performTSNE()

        clf = self.pickClassifier()
        prediction, score = [], []
        if applyPCA:
            if self.printing:
                print('Optimizing PCA components')
            while pca_comp < len(self.features_train[0]):
                prediction_n, score_n = self.optimizePCA(pca_comp, clf)
                # prediction_n1, score_n1 = self.optimizePCA(pca_comp + 1, clf)
                if self.printing:
                    print('Current {0} components score: {1}'.format(pca_comp, score_n))
                # print('Current {0} components score: {1}'.format(pca_comp + 1, score_n1))
                if score_n >= 0.8:  # np.round(score_n, 2) == np.round(score_n1, 2):
                    prediction, score = prediction_n, score_n
                    if self.printing:
                        print('Optimal component number is: {}'.format(pca_comp))
                    break
                pca_comp = pca_comp + 1
        elif applyTSNE:
            if self.printing:
                print('Optimizing by TSNE')
            tsne_train, tsne_test = self.performTSNE()
            prediction, score = self.performClassification(clf, tsne_train, self.labels_train, tsne_test, self.labels_test)
        else:
            prediction, score = self.performClassification(clf, self.features_train, self.labels_train, self.features_test, self.labels_test)
        if self.printing:
            self.printClassifierResults(prediction, ['Normal', 'AD'])

        return score
