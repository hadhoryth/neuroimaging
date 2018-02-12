import pandas as pd
from os import listdir, path
import re
from itertools import count
import datetime
import numpy as np
"""
  Assumed folder structure
  Root --> Patient --> Modalities --> Measurement day --> Image
  Script generates overall .csv file related to the processed images
"""


def _get_internal_params(adni_csv, ptid, date):
    """
        Extracting basic parameters:
        **Age, MMSE, Gender, ADAS_11, ADAS_13**

        NOTE:
        dx_bl is in field `label`

        `Return` - dict()
    """
    patient = adni_csv[adni_csv.PTID.str.match(ptid)]
    days_range = (np.abs(patient.EXAMDATE - date) <
                  datetime.timedelta(days=365 // 2))
    patient_by_date = patient.loc[days_range == True]
    params = {
        'rid': patient_by_date.RID.values[0],
        'label': patient_by_date.DX_bl.values[0],
        'mmse': patient_by_date.MMSE.values[0],
        'age': np.floor(patient_by_date.AGE.values[0]),
        'gender': patient_by_date.PTGENDER.values[0],
        'adas_11': patient_by_date.ADAS11.values[0],
        'adas_13': patient_by_date.ADAS13.values[0]
    }
    return params


def create_csv(img_root=None, adni_csv_path=None):
    root = '/Users/XT/Documents/PhD/Granada/neuroimaging/csv_dataset'
    adni_csv = pd.read_csv(path.join(root, 'ADNI_MIXED.csv'))
    adni_csv.loc[:, 'EXAMDATE'] = adni_csv.loc[:, 'EXAMDATE'].apply(
        lambda d: datetime.datetime.strptime(d, r'%Y-%m-%d'))
    # adni_csv.to_csv(path.join(root, 'ADNI_MIXED.csv'))

    img_root = '/Volumes/ELEMENT/Alzheimer/ADNI_Rearranged'

    patients_info = dict()
    global_counter = count()
    for pat_id in sorted(listdir(img_root)):
        subpath = path.join(img_root, pat_id)
        for modality in listdir(subpath):
            exam_date = path.join(subpath, modality)
            current_modality = 'PET'
            pattern = '^wrNewOrigin.*nii$'
            if re.match(r'^Coreg', modality):
                current_modality = 'FDG'
            elif re.match(r'MT1', modality):
                current_modality = 'MRI'
                pattern = '^wADNI.*nii$'

            for date in listdir(exam_date):
                img_fld = path.join(exam_date, date)
                d_datetime = datetime.datetime.strptime(
                    date[:-2], r'%Y-%m-%d_%H_%M_%S')
                for img in listdir(img_fld):
                    if re.match(pattern, img):
                        k = next(global_counter)
                        patients_info[k] = {'id': pat_id,
                                            'modality': current_modality,
                                            'date': d_datetime}
                        patients_info[k].update(_get_internal_params(adni_csv, pat_id, d_datetime))
                        patients_info[k]['image'] = path.join(img_fld, img)

    frame = pd.DataFrame.from_dict(patients_info, orient='index')
    frame.to_csv('all_dataset.csv')
    print('Converting finshed!')


create_csv()
