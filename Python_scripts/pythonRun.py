import numpy as np
import mat_feature_extractor
from Analysis import Classifier
from Helper import Helpers
from colorama import init, Fore
init(autoreset=True)


def run_analysis(data, mode, printing=False):
    score = np.zeros(3)
    keys = ['normal', 'ad', 'labels_normal', 'labels_ad']
    clf = Classifier(print=printing)
    print('{0} classification: '.format(mode))
    print('Without dimensional reduction:')
    score[0] = clf.classify(data, keys)
    print('Accuracy: {0}'.format(score[0]))
    print('PCA')
    score[1] = clf.classify(data, keys, applyPCA=True)
    print('Accuracy: {0}'.format(score[1]))
    print('TSNE')
    score[2] = clf.classify(data, keys, applyTSNE=True)
    print('Accuracy: {0}'.format(score[2]))
    return score


def run_regular_split():
    accuracy_av45 = run_analysis(av45, 'AV45', printing=True)
    print('----------------------------------------------------')
    accuracy_fdg = run_analysis(fdg, 'FDG', printing=True)
    print('----------------------------------------------------')
    print('----------------------------------------------------')
    print(Fore.RED + '|', end="")
    print(Fore.GREEN + 'Total accuracies for AV45: {0}'.format(accuracy_av45), end="")
    print(Fore.RED + '|')
    print(Fore.RED + '|', end="")
    print(Fore.GREEN + 'Total accuracies for FDG: {0}'.format(accuracy_fdg), end="")
    print(Fore.RED + '|')


def run_kfold_split(data):
    classifier = Classifier(print=False)
    classifier.data = data
    classifier.generateTrainTestSet(0, 0)
    clf = classifier.pickClassifier()
    print(clf)
    scores = classifier.classifyWithKFold(clf)
    return scores


if __name__ == '__main__':
    home = '/Users/XT/Documents/PhD/Granada/neuroimaging/ADNI_mat'
    # home = '/Volumes/ELEMENT/Alzheimer/ADNI_mat'
    av45, av45_dx, fdg, fdg_dx = mat_feature_extractor.extract_features(home)
    # Save variable for debug purposes
    # import pickle
    # with open('objs.pickle', 'wb') as f:
    #     pickle.dump([av45_dx, fdg_dx], f)

    scores = run_kfold_split(av45)

    hlp = Helpers()
    print('AV45 statistics:')
    hlp.printStatistics(av45_dx['normal'], av45_dx['ad'])
    print('Mean score for 10 folds: {0}'.format(scores.mean()))

    print('-----------------------------------------')

    print('FDG statistics:')
    hlp.printStatistics(fdg_dx['normal'], fdg_dx['ad'])
    scores = run_kfold_split(fdg)
    print('Mean score for 10 folds: {0}'.format(scores.mean()))
