import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from nipype.interfaces import spm
import sys
sys.path.append('/Users/XT/Documents/PhD/Granada/neuroimaging/Python_scripts')
import n_utils as ut
import h5py as h5
import time


def convert_label(label, n=3):
    """
        Possible options for labels are:\n
        **CN** - control normal --> `label`: 0\n
        **LMCI** - late  mild cognitive impairment --> `label`: 1\n
        **EMCI** - early mild cognitive impairment --> `label`: 1\n
        **AD** - Alzheimer's disease --> `label`: 2\n

        Return
        ------
        `hot_value` of the label
    """
    def generate_hot(pos):
        orig = np.zeros(n)
        orig[pos] = 1
        return orig
    conversion_table = dict(CN=0, LMCI=1, EMCI=1, AD=2)
    return generate_hot(conversion_table.get(label, 0))


def create_dataset(*, modality='PET', filename='generated.pickle'):
    """
        Csv file consists of columns:\n
        `id`, `modality`, `date`, `rid`, `label`,
        `image`
    """
    adni_data = pd.read_csv('all_dataset.csv')
    image_subset_pet = adni_data[adni_data.modality == modality]

    image_count = len(image_subset_pet.image)
    print(f'Total number of images: {image_count}')
    image_shape = nib.load(image_subset_pet.image[0]).get_data().shape

    img_out_shape = (image_count, image_shape[0], image_shape[1], image_shape[2])
    lab_out_shape = (image_count, convert_label(image_subset_pet.label[0]).shape[0])

    with h5.File(filename, 'w') as f:
        features = f.create_dataset('features', shape=img_out_shape)
        labels = f.create_dataset('labels', shape=lab_out_shape)
        for i, (path, label) in enumerate(zip(image_subset_pet.image, image_subset_pet.label)):
            features[i] = nib.load(path).get_data()
            labels[i] = convert_label(label)
            ut.print_progress(i, image_count)

    print(f'Data saved to {filename}')


def reslice_volume(volume, labels, slices, filename):
    orig_shape = volume.shape
    f_shape = (orig_shape[0], orig_shape[1], orig_shape[2], slices[1] - slices[0], 1)

    with h5.File(filename, 'w') as f:
        features = f.create_dataset('features', shape=f_shape)
        labels = f.create_dataset('labels', data=labels)
        for i in range(orig_shape[0]):
            z = ut.check_for_nans(volume[i][:, :, slices[0]: slices[1]])
            z[z < 0.0001] = 0.0
            features[i] = z.reshape(z.shape[0], z.shape[1], z.shape[2], 1)
    print('done')


if __name__ == '__main__':
    adni_data = h5.File('/Volumes/ELEMENT/Alzheimer/generated.h5', 'r')
    reslice_volume(adni_data['features'], adni_data['labels'], [7, 74],
                   '/Volumes/ELEMENT/Alzheimer/zeromin_re_7_74_generated.h5')
