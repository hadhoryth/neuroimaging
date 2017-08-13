import pickle
import os.path
import numpy as np
import mat_feature_extractor
from data_learn import Analysis
from Helper import Helpers
from sklearn.metrics import accuracy_score


def getData(mat_home, useCache=True):
    cached_data = 'formatted_data.pickle'
    if useCache and os.path.isfile(cached_data):
        with open(cached_data, 'rb') as r_bytes:
            return pickle.load(r_bytes)
    else:
        av45, av45_dx, fdg, fdg_dx = mat_feature_extractor.extract_features(mat_home)
        with open(cached_data, 'wb') as w_bytes:
            pickle.dump([av45, av45_dx, fdg, fdg_dx], w_bytes)
        return av45, av45_dx, fdg, fdg_dx


def printStatistics(hlp, mode, stat_data, scores):
    print('-----------------------------------------')
    print('{0} statistics'.format(mode))
    hlp.printStatistics(stat_data)
    print('Best mean score for 10 folds: {0}'.format(scores))
    print('-----------------------------------------')


if __name__ == '__main__':
    mat_home = '/Users/XT/Documents/PhD/Granada/neuroimaging/ADNI_mat'
    hlp = Helpers()
    keys = ['normal', 'mci', 'ad', 'labels_normal', 'labels_mci', 'labels_ad']
    data_av45, data_fdg = hlp.extractFearutes(mat_home, keys)

    split = [0, 0]

    analyzer = Analysis()
    print('AV45 processing')
    # # scores, pred_data = analizer.classify(av45, keys, 'svm', training_split=split, clf_cache_name='av45')
    scores = analyzer.classify(data_av45['av45'], keys, 'svm', training_split=split, clf_cache_name='av45')
    printStatistics(hlp, 'AV45', [data_av45['dx']['normal'], data_av45['dx']['mci'], data_av45['dx']['ad']], scores)
    # # r = accuracy_score(np.asarray(pred_data[0][0]), np.asarray(pred_data[1][0]))

    print('FDG processing')
    # scores, pred_data = analizer.classify(fdg, keys, 'svm', training_split=split, clf_cache_name='fdg')
    scores = analyzer.classify(data_fdg['fdg'], keys, 'svm', training_split=split, clf_cache_name='fdg')
    printStatistics(hlp, 'FDG', [data_fdg['dx']['normal'], data_fdg['dx']['mci'], data_fdg['dx']['ad']], scores)
