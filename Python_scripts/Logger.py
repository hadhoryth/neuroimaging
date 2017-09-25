from colorama import init, Fore
init(autoreset=True)
from datetime import datetime
import numpy as np


class Log:
    @staticmethod
    def info(tag, message):
        tag = tag.lower().title()
        print(
            Fore.GREEN + '[{0} {1}]:  '.format(datetime.now().strftime('%H:%M:%S'), tag), end='')
        print(message)

    @staticmethod
    def warning(tag, message):
        tag = tag.lower().title()
        print(
            Fore.RED + '[{0} {1}]:  '.format(datetime.now().strftime('%H:%M:%S'), tag), end='')
        print(Fore.RED + message)

    @staticmethod
    def classifier_details(tag, clf):
        print(
            Fore.GREEN + '[{0} {1}]:  '.format(datetime.now().strftime('%H:%M:%S'), tag.upper()))
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, param in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, param))
        print('Print best estimator: ', clf.best_params_, end='')
        print('   Score: ', clf.best_score_)
        print('--------------------------------------------------')

    @staticmethod
    def cross_val_details(c, gamma, scores):
        tag = 'cross validation'.title()
        print(
            Fore.GREEN + '[{0} {1}]:  '.format(datetime.now().strftime('%H:%M:%S'), tag))
        print('C:', c, '    Mean_C: ', np.mean(c))
        print('Gamma:', gamma, '    Mean_Gamma: ', np.mean(gamma))
        print('Scores: {0}'.format(scores),
              '    Mean_Scores: {0:0.3f}'.format(np.mean(scores)))
        print('Selected C: ',  np.dot(np.mean(c), np.mean(scores).T),
              ' Gamma: ', np.dot(np.mean(gamma), np.mean(scores).T))
