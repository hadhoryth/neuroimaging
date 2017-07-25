from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


class Helpers:
    def fancy_plot_confusion_matrix(cm, target_names, title='Confusion matrix',
                                    cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        plt.tight_layout()

        width, height = cm.shape

        for x in range(width):
            for y in range(height):
                plt.annotate(str(cm[x][y]), xy=(y, x),
                             horizontalalignment='center',
                             verticalalignment='center')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def checkMMSE(self, mmse):
        if np.isnan(mmse):
            return 0
        else:
            return mmse

    def getStatistics(self, patients_data):
        # Gender : Male - 1, Female - 2
        male_age, male_mmse = [], []
        female_age, female_mmse = [], []
        for patient in patients_data:
            if patient['gender'] == 'Male':
                male_age.append(patient['age'])
                male_mmse.append(self.checkMMSE(patient['mmse']))
            else:
                female_age.append(patient['age'])
                female_mmse.append(self.checkMMSE(patient['mmse']))

        male_age, male_mmse = np.asarray(male_age), np.asarray(male_mmse)
        female_age, female_mmse = np.asarray(female_age), np.asarray(female_mmse)

        stat = {'total': len(patients_data),
                'av_male': np.mean(male_age), 'std_male': np.std(male_age), 'av_male_mmse': np.mean(male_mmse),
                'av_female': np.mean(female_age), 'std_female': np.std(female_age), 'av_female_mmse': np.mean(female_mmse)}

        return np.array([stat['total'], stat['av_male'], stat['av_female'], stat['std_male'], stat['std_female'], stat['av_male_mmse'],
                         stat['av_female_mmse']]).reshape(1, 7)

    def printStatistics(self, normal, ad):
        stat_n = self.getStatistics(normal)
        stat_ad = self.getStatistics(ad)
        statistics = np.vstack([stat_n, stat_ad])

        types = ['Normal', 'AD']
        params = ['Total', 'Av_Male', 'Av_Female', 'Std Male', 'Std Female', 'AV_Male MMSE', 'AV_Female MMSE']
        results = pd.DataFrame(statistics, types, params)
        print(results)
