from features_learn import Features
from data_learn import Analysis
from Helper import Helpers
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from models import NeuralNetwork


def generateDict(normal, mci, ad):
    labels_norm = np.zeros(len(normal))
    labels_mci = np.ones(len(mci))
    labels_ad = np.ones(len(ad)) + 1
    return {'normal': normal, 'mci': mci, 'ad': ad,
            'labels_normal': labels_norm.T, 'labels_mci': labels_mci.T,
            'labels_ad': labels_ad.T}


def printTuniningDetails(clf):
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print('Print best estimator: ', clf.best_params_, end='')
    print('   Score: ', clf.best_score_)
    print('--------------------------------------------------')


def runAnalysis(all_data, classifier='svm', printDetails=True, params=None):
    # all_data has fields: normal, ad, mci
    # Create folds for every patient type
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=23)
    features, labels = np.asarray(
        all_data[0]), np.asarray(all_data[1], dtype=int)

    # Get Classifier
    if params is None:
        # Estimate parameters with GridSearchCV
        clf = als.getClassifier(classifier)
    else:
        # Set parameters
        if classifier == 'svm':
            clf = SVC(C=params['C'], gamma=params['Gamma'],
                      kernel='rbf', decision_function_shape='ovr')
        elif classifier == 'lgr':
            clf = LogisticRegression(C=params['C'], solver='newton-cg')

            # Ininitializing numpy arrays for storing classifier params for every fold
    _C, _Gamma, _Scores, _Accuracies = np.array(
        []), np.array([]), np.array([]), np.array([])
    # Loop through all folds
    for train_idx, test_idx in skf.split(features, labels):

        scaler = StandardScaler()
        scaled_features_train = scaler.fit_transform(features[train_idx])

        # Train classifier
        clf.fit(scaled_features_train, labels[train_idx])
        if params is None:
            if printDetails:
                printTuniningDetails(clf)
            _C = np.append(_C, clf.best_params_['C'])
            if classifier == 'svm':
                _Gamma = np.append(_Gamma, clf.best_params_['gamma'])
            _Scores = np.append(_Scores, clf.best_score_)
        else:
            scaled_features_test = scaler.transform(features[test_idx])
            predictions = clf.predict(scaled_features_test)
            _Accuracies = np.append(_Accuracies, accuracy_score(
                predictions, labels[test_idx]))
            print('Accuracy score: ', accuracy_score(
                predictions, labels[test_idx]))

    if params is None:
        print('========================= Results ==========================')
        print('C:', _C, '    Mean_C: ', np.mean(_C))
        print('Gamma:', _Gamma, '    Mean_Gamma: ', np.mean(_Gamma))
        print('Scores: ', _Scores, '    Mean_Scores: ', )
        print('Selected C: ',  np.dot(np.mean(_C), np.mean(_Scores).T),
              ' Gamma: ', np.dot(np.mean(_Gamma), np.mean(_Scores).T))
        return {'C': np.dot(np.mean(_C), np.mean(_Scores).T), 'Gamma': np.dot(np.mean(_Gamma), np.mean(_Scores).T)}

    return _Accuracies


if __name__ == '__main__':
    # Reading data, Inintializing classes in the workspace
    hlp, ft, als = Helpers(), Features(), Analysis()
    all_data = hlp.saveReadToLocal('read', '_all_av45', None, '_cache')
    all_data = all_data['av45']
    keys = ['normal', 'mci', 'ad', 'labels_normal', 'labels_mci', 'labels_ad']

    mixed_all_data = als.getTrainingSet(all_data, 0, 0, keys, do_split=False)
    print('Features dimention before normalization: ',
          len(mixed_all_data[0][0]))
    mixed_all_data = ft.normalizeFeatures(mixed_all_data)
    print('Features dimention after normalization: ', len(mixed_all_data[0][0]))
    # mixed_all_data = als._cleanDataset(mixed_all_data[0],  mixed_all_data[1])

    # grid_params = runAnalysis(mixed_all_data, classifier='svm')
    # a = runAnalysis(mixed_all_data, params=grid_params, classifier='svm')
    # print('Overall accuracy: ', np.mean(a))

    input_size = len(mixed_all_data[0][0])
    nn = NeuralNetwork([input_size, input_size, 40, 40, 3])
    input_data = np.asarray(mixed_all_data[0]).reshape(
        (input_size, len(mixed_all_data[0])))
    input_labels = np.asarray(mixed_all_data[1]).reshape(
        (1, len(mixed_all_data[1])))
    nn.run(input_data, input_labels, 0.01, 5000)

    test = np.asarray(mixed_all_data[0][0]).reshape(
        (input_size, 1))
    print(nn.predict(test))
