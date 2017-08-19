import pickle
import os.path
from data_learn import Analysis
from features_learn import Features
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
    print('\n================= Total {0} statistics ========================'.format(mode))
    hlp.printStatistics(stat_data)
    print('Best mean score for 10 folds: {0}'.format(scores))
    print('===============================================================')


def getFeatures(mode, keys, normilize=False, analyzer=Analysis(), hlp=Helpers()):
    cached_info_names = ['_data_av45', '_data_fdg']
    if hlp.cacheExist(cached_info_names[0], cache_path) and hlp.cacheExist(cached_info_names[1], cache_path):
        print('Loading dataset from cached files ........ ', end='')
        data_av45 = hlp.saveReadToLocal(mode, cached_info_names[0], None, cache_path)
        data_fdg = hlp.saveReadToLocal(mode, cached_info_names[1], None, cache_path)
        print('OK')
    else:
        print('Extracting features.....-> ', end='')
        mode = 'write'
        data_av45, data_fdg = hlp.extractFearutes(mat_home, keys)
        if normilize:
            f = Features()
            data_av45['av45'] = f.normalizeFeatures(data_av45['av45'])
            data_fdg['fdg'] = f.normalizeFeatures(data_fdg['fdg'])

        def updateCache(data, info, name):
            train, test, labels_train, labels_test = analyzer.getTrainingSet(data, 0.1, 30, keys)
            out_dict = {'data': {'train': train, 'test': test,
                                 'labels_train': labels_train, 'labels_test': labels_test}, 'info': info}
            hlp.saveReadToLocal(mode, name, out_dict, cache_path)
            return out_dict

        print('saving features.... ', end='')
        data_av45 = updateCache(data_av45['av45'], data_av45['dx'], cached_info_names[0])
        data_fdg = updateCache(data_fdg['fdg'], data_fdg['dx'], cached_info_names[1])
        print('OK')

    return data_av45, data_fdg


def runAnalysis(data, data_dx, keys, clf_type='svm', img_type='Unknown', analyzer=Analysis(),
                hlp=Helpers()):
    split = [0, 0]
    print('================== {0} processing ======================'.format(img_type))
    clf_name = '_' + img_type.lower() + '_' + clf_type
    scores, cm = analyzer.classify(
        data, keys, clf_type, training_split=split, clf_cache_name=clf_name)
    printStatistics(hlp, img_type, data_dx, scores)
    hlp.fancy_plot_confusion_matrix(
        cm, ['Normal', 'MCI', 'AD'], title=img_type + ' confusion matrix')


if __name__ == '__main__':
    cache_path = '_cache'
    mat_home = '/Users/XT/Documents/PhD/Granada/neuroimaging/ADNI_mat'
    keys = ['normal', 'mci', 'ad', 'labels_normal', 'labels_mci', 'labels_ad']

    data_av45, data_fdg = getFeatures('read', keys, normilize=True)

    av45_dx = [data_av45['info']['normal'], data_av45['info']['mci'], data_av45['info']['ad']]
    runAnalysis(data_av45['data'], av45_dx, keys, img_type='AV45')

    fdg_dx = [data_fdg['info']['normal'], data_fdg['info']['mci'], data_fdg['info']['ad']]
    runAnalysis(data_fdg['data'], fdg_dx, keys, img_type='FDG')
