from features_learn import Features
from data_learn import Analysis
from Helper import Helpers
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


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


def _processFeatures(normal, mci, ad, normal_idx=None, mci_idx=None, ad_idx=None,
                     normal_scaler=None, mci_scaler=None, ad_scaler=None):
    keys = ['normal', 'mci', 'ad', 'labels_normal', 'labels_mci', 'labels_ad']
    # Convert lists to numpy array
    normal, mci, ad = np.asarray(normal), np.asarray(mci), np.asarray(ad)

    # Get dictionary data for thefold
    features_dict = generateDict(normal[normal_idx], mci[mci_idx], ad[ad_idx])

    # Normilize features by removing outlayers in every brain region
    normilized_features_dict = ft.normalizeFeatures(features_dict)

    mode = 'test'
    if normal_scaler is None:
        mode = 'train'

    # Initializing Scalers
    if mode == 'train':
        normal_scaler, mci_scaler, ad_scaler = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
        scaled_normal = normal_scaler.fit_transform(normilized_features_dict['normal'])
        scaled_mci = mci_scaler.fit_transform(normilized_features_dict['mci'])
        scaled_ad = ad_scaler.fit_transform(normilized_features_dict['ad'])
    else:
        scaled_normal = normal_scaler.transform(normilized_features_dict['normal'])
        scaled_mci = mci_scaler.transform(normilized_features_dict['mci'])
        scaled_ad = ad_scaler.transform(normilized_features_dict['ad'])

    scaled_features_dict = generateDict(scaled_normal, scaled_mci, scaled_ad)

    # Concatinating all the data for future classification
    features_mixed, labels_mixed = als.getTrainingSet(
        scaled_features_dict, 0, 0, keys, do_split=False)
    if mode == 'train':
        return features_mixed, labels_mixed, normal_scaler, mci_scaler, ad_scaler
    else:
        return features_mixed, labels_mixed


def runAnalysis(all_data, printDetails=True, params=None):
    # all_data has fields: normal, ad, mci
    # Create folds for every patient type
    skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=23)
    split_norm = skf.split(all_data['normal'], all_data['labels_normal'])
    split_mci = skf.split(all_data['mci'], all_data['labels_mci'])
    split_ad = skf.split(all_data['ad'], all_data['labels_ad'])

    # Get Classifier
    if params is None:
        # Estimate parameters with GridSearchCV
        clf = als.getClassifier('svm')
    else:
        # Set parameters
        clf = SVC(C=params['C'], gamma=params['Gamma'], kernel='rbf', decision_function_shape='ovr')

    # Ininitializing numpy arrays for storing classifier params for every fold
    _C, _Gamma, _Scores, _Accuracies = np.array([]), np.array([]), np.array([]), np.array([])
    # Loop through all folds
    for (train_norm_i, test_norm_i), (train_mci_i, test_mci_i), (train_ad_i, test_ad_i) in zip(split_norm, split_mci, split_ad):

        features_mixed, labels_mixed, n_scaler, m_scaler, a_scaler = _processFeatures(
            all_data['normal'], all_data['mci'], all_data['ad'], normal_idx=train_norm_i, mci_idx=train_mci_i, ad_idx=train_ad_i)

        # Train classifier
        clf.fit(features_mixed, labels_mixed)
        if params is None:
            if printDetails:
                printTuniningDetails(clf)
            _C = np.append(_C, clf.best_params_['C'])
            _Gamma = np.append(_Gamma, clf.best_params_['gamma'])
            _Scores = np.append(_Scores, clf.best_score_)
        else:
            test_features_mixed, test_labels_mixed = _processFeatures(
                all_data['normal'], all_data['mci'], all_data['ad'], normal_idx=test_norm_i, mci_idx=test_mci_i, ad_idx=test_ad_i,
                normal_scaler=n_scaler, mci_scaler=m_scaler, ad_scaler=a_scaler)
            predictions = clf.predict(test_features_mixed)
            _Accuracies = np.append(_Accuracies, accuracy_score(predictions, test_labels_mixed))
            print('Accuracy score: ', accuracy_score(predictions, test_labels_mixed))

    if params is None:
        print('========================= Results ==========================')
        print('C:', _C, '    Mean_C: ', np.mean(_C))
        print('Gamma:', _Gamma, '    Mean_Gamma: ', np.mean(_Gamma))
        print('Scores: ', _Scores, '    Mean_Scores: ', np.mean(_Scores))
        return {'C': np.mean(_C), 'Gamma': np.mean(_Gamma)}

    return _Accuracies


if __name__ == '__main__':
    # Reading data, Inintializing classes in the workspace
    hlp, ft, als = Helpers(), Features(), Analysis()
    all_data = hlp.saveReadToLocal('read', '_all_av45', None, '_cache')
    all_data = all_data['av45']
    keys = ['normal', 'mci', 'ad', 'labels_normal', 'labels_mci', 'labels_ad']

    grid_params = runAnalysis(all_data)
    runAnalysis(all_data, params=grid_params)
