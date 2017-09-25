import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc, make_scorer, f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgb
from Logger import Log

from colorama import init, Fore
init(autoreset=True)


class Analysis:

    def _get_classifier(self, type, k=3):
        if type.lower() == 'knn':
            Log.info('Model info', 'Nearest Neighbor Classifier has been selected')
            clf = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')
        elif type.lower() == 'svm':
            Log.info('Model info', 'Support vector machine has been selected')
            tuned_param = {'C': [0.1, 10, 10, 50, 100],
                           'gamma': [0.001, 0.01, 0.1, 0.2]}
            clf = GridSearchCV(SVC(kernel='rbf', decision_function_shape='ovr',
                                   random_state=0), tuned_param, cv=5)
        elif type.lower() == 'lin_svm':
            Log.info('Model info', 'Linear Support Vector machine has been selected')
            clf = OneVsRestClassifier(LinearSVC(random_state=0, C=5))
        elif type.lower() == 'lgr':
            Log.info('Model info', 'Logistic regression has been selected')
            tuned_param = {'C': [0.01, 0.1, 10, 10, 100,
                                 200], 'solver': ['newton-cg', 'lbfgs']}
            f1_scorer = make_scorer(f1_score, pos_label='0')
            clf = GridSearchCV(LogisticRegression(),
                               param_grid=tuned_param, cv=5)
        elif type.lower() == 'xgb':
            tuned_param = {'max_depth': [5],
                           'n_estimators': [20, 300],
                           'learning_rate': [0.01, 0.4],
                           'silent': [1.0],
                           'gamma': [0.01, 0.1, 0.5]}
            model = xgb.XGBClassifier(seed=42, objective='multi:softmax')
            clf = GridSearchCV(model, param_grid=tuned_param, cv=5)
        return clf

    def do_split(self, features, labels, test=0.1):
        return train_test_split(
            features, labels, test_size=test, random_state=42)

    def mix_data(self, data_dict, keys, logging=False, test=None):
        step = len(keys) // 2
        features, labels = data_dict[keys[0]], data_dict[keys[step]]
        for key in range(1, step):
            features = np.concatenate((features, data_dict[keys[key]]))
            labels = np.concatenate((labels, data_dict[keys[key + step]]))

        X_train, X_test, y_train, y_test = self.do_split(
            features, labels, test)
        if logging:
            Log.info(
                'split', 'Train size - {0}; Test size - {1}'.format(len(y_train), len(y_test)))
            Log.info('split', 'Train:: Normal - {0}; MCI - {1}; AD - {2}'.format(len(
                y_train[y_train == 0]), len(y_train[y_train == 1]), len(y_train[y_train == 2])))
            Log.info('split', 'Test:: Normal - {0}; MCI - {1}; AD - {2}'.format(len(
                y_test[y_test == 0]), len(y_test[y_test == 1]), len(y_test[y_test == 2])))
        out_dict = {'train': {'features': X_train, 'labels': y_train},
                    'test': {'features': X_test, 'labels': y_test}}

        return out_dict

    def train_model(self, clf_type, features, labels, params=None, logging=False):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=23)
        # params used for setting model after training
        if params is None:
            clf = self._get_classifier(clf_type)
        else:
            if clf_type == 'svm':
                clf = SVC(C=params['C'], gamma=params['Gamma'],
                          kernel='rbf', decision_function_shape='ovr')
            elif clf_type == 'lgr':
                clf = LogisticRegression(C=params['C'], solver='newton-cg')

        _C, _Gamma, _Scores, _Accuracies = np.array(
            []), np.array([]), np.array([]), np.array([])
        selected_params = dict()
        for train_idx, test_idx in skf.split(features, labels):
            # if clf_type == 'xgb':
            #     if logging:
            #         watchlist = [(dfeatures, 'train')]
            #     dfeatures = xgb.DMatrix(
            #         features[train_idx], labels=labels[train_idx])
            #     bst = xgb.train(clf, dfeatures, watchlist, 5)
            # else:
            clf.fit(features[train_idx], labels[train_idx])
            if params is None:
                if logging:
                    Log.classifier_details(clf_type, clf)
                if clf_type == 'svm':
                    _C = np.append(_C, clf.best_params_['C'])
                    _Gamma = np.append(_Gamma, clf.best_params_['gamma'])
                _Scores = np.append(_Scores, clf.best_score_)
            else:
                predictions = clf.predict(features[test_idx])
                _Accuracies = np.append(_Accuracies, accuracy_score(
                    predictions, labels[test_idx]))
                print('Accuracy score: ', accuracy_score(
                    predictions, labels[test_idx]))

        if logging and params is None:
            Log.cross_val_details(_C, _Gamma, _Scores)
        if params is None:
            selected_params = {'C': np.dot(np.mean(_C), np.mean(
                _Scores).T), 'Gamma': np.dot(np.mean(_Gamma), np.mean(_Scores).T)}

        return _Accuracies, selected_params
