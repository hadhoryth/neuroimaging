import scipy.io as sio
import numpy as np
from os import listdir, path
from colorama import init


def getFolderContents(fld_path):
    fld = listdir(fld_path)
    hidden_folder = '.DS_Store'
    for i in range(0, len(fld)):
        if fld[i] == hidden_folder:
            fld.remove(fld[i])
            return fld
    return fld


def packFeatures(a, b, keys):
    labels_a = np.zeros(len(a)) + 1
    labels_b = np.zeros(len(b)) + 2

    return {keys[0]: np.asarray(a), keys[1]: np.asarray(b),
            keys[2]: labels_a, keys[3]: labels_b}


def convertToDict(data, fld):
    keys = ['id', 'age', 'gender', 'mmse', 'dx_change']
    if not data['id']:
        print(fld)
        return None
    info = {}
    for i in range(len(keys)):
        info[keys[i]] = data[keys[i]][0][0]

    return info


def extract_features(files_dir):
    init(autoreset=True)
    features_AD_AV45, features_AD_AV45_DX = [], []
    features_AD_FDG, features_AD_FDG_DX = [], []
    features_Normal_AV45, features_Normal_AV45_DX = [], []
    features_Normal_FDG, features_Normal_FDG_DX = [], []

    patients_folder = getFolderContents(files_dir)
    for fld in patients_folder:
        patient_type = 'Normal'
        if fld == 'AD':
            patient_type = 'AD'
        img_flds_path = path.join(files_dir, patient_type)
        img_flds_contents = getFolderContents(img_flds_path)
        for sub_fld in img_flds_contents:
            img_type, load_var = 'AV45', 'brainRegions_pet'
            if sub_fld == 'FDG':
                img_type, load_var = 'FDG', 'brainRegions_fdg'
            features_path = path.join(img_flds_path, img_type)
            files_path = getFolderContents(path.join(img_flds_path, img_type))

            for file in files_path:
                mat_contents = sio.loadmat(path.join(features_path, file))
                brain_features = mat_contents[load_var][0]
                dx_data = convertToDict(mat_contents['dx_data'][0], file)

                if patient_type is 'Normal':
                    if img_type is 'FDG':
                        features_Normal_FDG.append(brain_features)
                        features_Normal_FDG_DX.append(dx_data)
                    else:
                        features_Normal_AV45.append(brain_features)
                        features_Normal_AV45_DX.append(dx_data)
                elif patient_type is 'AD':
                    if img_type is 'FDG':
                        features_AD_FDG.append(brain_features)
                        features_AD_FDG_DX.append(dx_data)
                    else:
                        features_AD_AV45.append(brain_features)
                        features_AD_AV45_DX.append(dx_data)
    dict_keys = ['normal', 'ad', 'labels_normal', 'labels_ad']
    AV45 = packFeatures(features_Normal_AV45, features_AD_AV45, dict_keys)
    AV45_DX = packFeatures(features_Normal_AV45_DX, features_AD_AV45_DX, dict_keys)

    FDG = packFeatures(features_Normal_FDG, features_AD_FDG, dict_keys)
    FDG_DX = packFeatures(features_Normal_FDG_DX, features_AD_FDG_DX, dict_keys)

    return AV45, AV45_DX, FDG, FDG_DX
