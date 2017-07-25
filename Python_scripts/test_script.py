import pickle
import numpy as np
import pandas as pd


def checkMMSE(mmse):
    if np.isnan(mmse):
        return 0
    else:
        return mmse


def getStatistics(patients_data):
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
    female_age, female_mmse = np.asarray(female_age), np.asarray(female_mmse)

    stat = {'total': len(patients_data),
            'av_male': np.mean(male_age), 'std_male': np.std(male_age), 'av_male_mmse': np.mean(male_mmse),
            'av_female': np.mean(male_age), 'std_female': np.std(male_age), 'av_female_mmse': np.mean(male_mmse)}

    return np.array([stat['total'], stat['av_male'], stat['av_female'], stat['std_male'], stat['std_female'], stat['av_male_mmse'],
                     stat['av_female_mmse']]).reshape(1, 7)


with open('objs.pickle', 'rb') as f:
    av45, fdg = pickle.load(f)

stat_n = getStatistics(av45['normal'])
stat_ad = getStatistics(av45['ad'])
statistics = np.vstack([stat_n, stat_ad])


types = ['Normal', 'AD']
params = ['Total', 'Av_Male', 'Av_Female', 'Std Male', 'Std Female', 'AV_Male MMSE', 'AV_Female MMSE']
results = pd.DataFrame(statistics, types, params)
print(results)
