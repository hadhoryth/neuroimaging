"""
Module for every helper method needed for preparing, analysing AD
"""
from sklearn.model_selection import train_test_split
from Logger import Log
import numpy as np
import pickle
import sys
from numba import jit
import h5py as h5


def dump_to_fld(file_path, var):
    """ Saves variable to file
    Parameters
    ------
    file_path: where to save which includes extension `filename.pickle`
    var: actual variable
    """
    with open(file_path, 'wb') as file_bytes:
        pickle.dump(var, file_bytes)


def read_from_fld(file_path):
    """ Reads variable from folder
    Parameters
    ------
    `file_path` to file with extension `*.pickle`
    Return
    ------
    A `list` of varibles saved in the file    
    """
    with open(file_path, 'rb') as file_bytes:
        return pickle.load(file_bytes)


def dump_to_h5(file_path, var, *, var_name='default'):
    with h5.File(file_path, 'w') as f:
        dset = f.create_dataset(var_name, var.shape, 'float', var)


def reaf_from_h5(file_path):
    return h5.File(file_path, 'r')


@jit
def mix_data(data, keys, logging=False, test_frac=0.1):
    """
       Method to pack all data and split it in the train and test sets
       Output forma is dict with – "train" and "test" fields
    """
    assert isinstance(data, dict), 'Data has to be a dictionary'
    assert isinstance(keys, list), 'Keys has to be a list'

    step = len(keys) // 2
    features, labels = data[keys[0]], data[keys[step]]
    for key in range(1, step):
        features = np.concatenate((features, data[keys[key]]))
        labels = np.concatenate((labels, data[keys[key + step]]))
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_frac, random_state=42)
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


@jit
def concat_dicts(dict1, dict2):
    assert dict1.keys() == dict2.keys(), 'Dictionaries keys must be the same'
    result = dict()
    for key, value in dict1.items():
        if len(value.shape) == 1:
            result[key] = np.concatenate([value, dict2[key]])
        else:
            result[key] = np.vstack([value, dict2[key]])
    return result


@jit
def check_for_duplicates(data_a, data_b, label=''):
    matched_elements = np.asarray([], dtype=int)
    for i in range(len(data_a)):
        for j in range(len(data_b)):
            if i != j and np.array_equal(data_a[i], data_b[j]):
                matched_elements = np.append(matched_elements, i)
                break
    if len(matched_elements) == 0:
        Log.info('Array checker', label + 'No dublicates found......OK!')
    else:
        Log.warning('Array ckecker', label +
                    'Duplicates found, check dataset!!')
        Log.info('Array checker', 'Duplicates: {0}'.format(matched_elements))


def get_init_data(file_path, show=False):
    """
    Reading data from file; Mixing all together and dividing into: training,
     testing and validation sets\n
    Return a dict with keys: `train, test, validation`
    """
    AD_KEYS = ['normal', 'mci', 'ad',
               'labels_normal', 'labels_mci', 'labels_ad']

    av45, fdg = read_from_fld(file_path)
    mixed_av45 = mix_data(av45['av45'], AD_KEYS, logging=show)
    mixed_fdg = mix_data(fdg['fdg'], AD_KEYS, logging=show)

    test_set = concat_dicts(mixed_av45['test'], mixed_fdg['test'])
    train, test, labels_train, labels_test = train_test_split(
        test_set['features'], test_set['labels'], test_size=0.5, random_state=42)
    dev_set = {'features': train, 'labels': labels_train}
    validation_set = {'features': test, 'labels': labels_test}

    mixed_av45['train']['features'] = np.vstack(
        [mixed_av45['train']['features'], mixed_fdg['train']['features']])
    mixed_av45['train']['labels'] = np.append(
        mixed_av45['train']['labels'], mixed_fdg['train']['labels'])

    check_for_duplicates(mixed_av45['train']
                         ['features'], dev_set['features'], 'Dev: ')
    check_for_duplicates(
        mixed_av45['train']['features'], validation_set['features'], 'Validation: ')

    return dict(train=mixed_av45, test=dev_set, validation=validation_set)


def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
        Works only with terminal run
    """
    bar_length = 60
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' %
                     (prefix, bar, percent, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def get_AD_stage_name(value):
    stages = {0: 'Normal', 1: 'MCI', 2: 'AD'}
    return stages.get(value, 'Unknown')


@jit
def check_for_nans(array, nan_value=0.01):
    for el in array:
        el[np.isnan(el)] = nan_value
    return array
