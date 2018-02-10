import itertools
import os.path
import pickle
from os import listdir, path

import numpy as np
import pandas as pd
import scipy.io as sio
from matplotlib import pyplot as plt

from Logger import Log


class Helpers:
    def getStatistics(self, patients_data):
        def checkMMSE(mmse):
            if np.isnan(mmse):
                return 0
            return mmse

        # Gender : Male - 1, Female - 2
        male_age, male_mmse = [], []
        female_age, female_mmse = [], []
        for patient in patients_data:
            if patient['gender'] == 'Male':
                male_age.append(patient['age'])
                male_mmse.append(checkMMSE(patient['mmse']))
            else:
                female_age.append(patient['age'])
                female_mmse.append(checkMMSE(patient['mmse']))

        male_age, male_mmse = np.asarray(male_age), np.asarray(male_mmse)
        female_age, female_mmse = np.asarray(
            female_age), np.asarray(female_mmse)

        stat = {'total': len(patients_data),
                'av_male': np.mean(male_age), 'std_male': np.std(male_age), 'av_male_mmse': np.mean(male_mmse),
                'av_female': np.mean(female_age), 'std_female': np.std(female_age), 'av_female_mmse': np.mean(female_mmse)}

        return np.array([stat['total'], stat['av_male'], stat['av_female'], stat['std_male'], stat['std_female'], stat['av_male_mmse'],
                         stat['av_female_mmse']]).reshape(1, 7)

    def printStatistics(self, data_dx):
        statistics = self.getStatistics(data_dx[0])
        for i in range(1, len(data_dx)):
            statistics = np.vstack([statistics, self.getStatistics(data_dx[i])])

        types = ['Normal', 'MCI', 'AD']
        params = ['Total', 'Av_Male', 'Av_Female', 'Std Male',
                  'Std Female', 'AV_Male MMSE', 'AV_Female MMSE']
        results = pd.DataFrame(statistics, types, params)
        print(results)

    def getFolderContents(self, fld_path):
        fld = listdir(fld_path)
        hidden_folder = '.DS_Store'
        for i in range(0, len(fld)):
            if fld[i] == hidden_folder:
                fld.remove(fld[i])
                return fld
        return fld

    def getLoadVariable(self, modality):
        if modality.lower() == 'av45':
            return 'brainRegions_pet'
        return 'brainRegions_fdg'

    def convertToDict(self, data):
        keys = ['id', 'age', 'gender', 'mmse', 'dx_change']
        if not data['id']:
            return None
        info = {}
        for i in range(len(keys)):
            info[keys[i]] = data[keys[i]][0][0]
        return info

    def initContainers(self):
        self.features_AD_AV45, self.features_AD_AV45_DX = [], []
        self.features_MCI_AV45, self.features_MCI_AV45_DX = [], []
        self.features_Normal_AV45, self.features_Normal_AV45_DX = [], []

        self.features_AD_FDG, self.features_AD_FDG_DX = [], []
        self.features_MCI_FDG, self.features_MCI_FDG_DX = [], []
        self.features_Normal_FDG, self.features_Normal_FDG_DX = [], []

    def updateFeatures(self, patient, modality, features, dx_data):
        def updateVaribale(what1, to1, what2, to2):
            what1.append(to1)
            what2.append(to2)

        # TODO Refactoring needed
        patient, modality = patient.lower(), modality.lower()
        if patient == 'normal':
            if modality == 'fdg':
                updateVaribale(self.features_Normal_FDG, features,
                               self.features_Normal_FDG_DX, dx_data)
            else:
                updateVaribale(self.features_Normal_AV45, features,
                               self.features_Normal_AV45_DX, dx_data)
        elif patient == 'mci':
            if modality == 'fdg':
                updateVaribale(self.features_MCI_FDG, features,
                               self.features_MCI_FDG_DX, dx_data)
            else:
                updateVaribale(self.features_MCI_AV45, features,
                               self.features_MCI_AV45_DX, dx_data)
        elif patient == 'ad':
            if modality == 'fdg':
                updateVaribale(self.features_AD_FDG, features,
                               self.features_AD_FDG_DX, dx_data)
            else:
                updateVaribale(self.features_AD_AV45, features,
                               self.features_AD_AV45_DX, dx_data)

    def packFeatures(self, keys):
        def formatFeatures(data, data1, gl_keys, in_keys):
            out_dict = {}
            step = len(keys) // 2
            for gl_key in gl_keys:
                out_dict.update({gl_key: {}})
                for key in range(step):
                    if gl_key != gl_keys[-1]:
                        out_dict[gl_key].update({keys[key]: data[key]})
                        out_dict[gl_key].update(
                            {keys[key + step]: np.zeros(len(data[key])) + key})
                    else:
                        out_dict[gl_key].update({keys[key]: data1[key]})

            return out_dict

        av45 = [self.features_Normal_AV45,
                self.features_MCI_AV45, self.features_AD_AV45]
        av45_dx = [self.features_Normal_AV45_DX,
                   self.features_MCI_AV45_DX, self.features_AD_AV45_DX]

        fdg = [self.features_Normal_FDG,
               self.features_MCI_FDG, self.features_AD_FDG]
        fdg_dx = [self.features_Normal_FDG_DX,
                  self.features_MCI_FDG_DX, self.features_AD_FDG_DX]

        return formatFeatures(av45, av45_dx, ['av45', 'dx'], keys), formatFeatures(fdg, fdg_dx, ['fdg', 'dx'], keys)

    def _extract_fearutes(self, root_directory, keys):
        # Getting folders with features: Normal, MCI, AD
        self.initContainers()
        root_contents = self.getFolderContents(root_directory)
        for fld in root_contents:
            modalities_path = path.join(root_directory, fld)
            modalities = self.getFolderContents(modalities_path)
            for mod in modalities:
                load_var = self.getLoadVariable(mod)
                patients_path = path.join(modalities_path, mod)
                patients_contents = self.getFolderContents(patients_path)
                for patient in patients_contents:
                    mat_contents = sio.loadmat(
                        path.join(patients_path, patient))
                    features = mat_contents[load_var][0]
                    dx_data = self.convertToDict(mat_contents['dx_data'][0])

                    self.updateFeatures(fld, mod.lower(), features, dx_data)
        return self.packFeatures(keys)

    def dump_to_local(self, file_path, var):
        with open(file_path, 'wb') as file_bytes:
            pickle.dump(var, file_bytes)

    @staticmethod
    def read_from_local(file_path):
        with open(file_path, 'rb') as file_bytes:
            return pickle.load(file_bytes)

    def get_features(self, root_info, name, cache_dir='_cache', logging=False):
        filename = os.path.join(cache_dir, name + '.pickle')
        if os.path.isfile(filename):
            if logging:
                Log.info('dump', 'Reading from {0}'.format(filename))
            return self.read_from_local(filename)
        info = self._extract_fearutes(
            root_info['mat_home'], root_info['keys'])
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)
        self.dump_to_local(filename, info)

        return info

    def fancy_plot_confusion_matrix(self, cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, plot=False):

        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        print(cm_norm)

        print('Confusion matrix, without normalization')
        print(cm)

        if plot:
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()

            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            fmt = '.2f'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.show()

    @classmethod
    def concat_dicts(self, dict1, dict2):
        """
            Dictionary keys must be the same
        """
        result = dict()
        for key, value in dict1.items():
            if len(value.shape) == 1:
                result[key] = np.concatenate([value, dict2[key]])
            else:
                result[key] = np.vstack([value, dict2[key]])
        return result
