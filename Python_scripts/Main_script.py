import pickle
import os.path
import mat_feature_extractor
from data_learn import Analysis
from Helper import Helpers


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


def printStatistics(hlp, mode, data_1, data_2, scores):
    print('-----------------------------------------')
    print('{0} statistics'.format(mode))
    hlp.printStatistics(data_1, data_2)
    print('Best mean score for 10 folds: {0}'.format(scores))
    print('-----------------------------------------')


if __name__ == '__main__':
    mat_home = '/Users/XT/Documents/PhD/Granada/neuroimaging/ADNI_mat'
    av45, av45_dx, fdg, fdg_dx = getData(mat_home)
    keys = ['normal', 'ad', 'labels_normal', 'labels_ad']

    split = [0, 0]
    hlp = Helpers()
    analizer = Analysis()
    print('AV45 processing')
    scores = analizer.classify(av45, keys, 'svm', training_split=split)
    printStatistics(hlp, 'AV45', av45_dx['normal'], av45_dx['ad'], scores)

    print('FDG processing')
    scores = analizer.classify(fdg, keys, 'svm', training_split=split)
    printStatistics(hlp, 'FDG', fdg_dx['normal'], fdg_dx['ad'], scores)
