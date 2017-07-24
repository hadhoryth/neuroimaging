import numpy as np
import mat_feature_extractor
from Analysis import Classifier
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

if __name__ == '__main__':
    home = '/Users/XT/Documents/MATLAB/ADNI_mat'
    av45, fdg = mat_feature_extractor.extract_features(home)

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
