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
# from keras import layers
# from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Conv2D, Flatten
# from keras.layers import AveragePooling3D, MaxPool2D, Dropout, GlobalAveragePooling3D, GlobalAveragePooling3D
# from keras.models import Model, Sequential


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


def model(input_shape):
    """
        Need to use Sequential()
    """
    # model = Sequential()
    # model.add(Input(input_shape))

    # model.add(Conv2D(40, (5, 5), strides=(1, 1)))
    # model.add(BatchNormalization(axis=3))
    # model.add(Activation('relu'))
    # model.add(MaxPool2D(pool_size=(3, 3)))

    # model.add(Conv2D(20, (7, 7), strides=(2, 2), padding='same'))
    # model.add(BatchNormalization(axis=3))
    # model.add(Activation('relu'))
    # model.add(MaxPool2D(pool_size=(3, 3)))

    # model.add(Dense(279, 'relu'))
    # model.add(Dense(40, 'relu'))
    # model.add(Dense(3, 'softmax'))

    X_input = Input(input_shape)
    X = ZeroPadding2D((2, 2))(X_input)

    X = Conv2D(80, (3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPool2D(pool_size=(3, 3))(X)

    X = Conv2D(60, (5, 5), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPool2D(pool_size=(3, 3))(X)

    X = Conv2D(40, (7, 7), strides=(2, 2), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPool2D(pool_size=(3, 3))(X)

    X = Flatten()(X)
    X = Dense(279, activation='relu')(X)
    X = Dense(100, activation='relu')(X)
    X = Dense(3, activation='softmax')(X)

    # # First layer CONV -> BN -> Relu
    # X = Conv2D(32, (7, 7), name='conv0')(X)
    # X = BatchNormalization(axis=3, name='bn0')(X)
    # X = Activation('relu')(X)

    # # Maxpooling
    # X = MaxPool2D(name='max_pool')(X)

    # # Output layer
    # X = Flatten()(X)
    # X = Dense(3, activation='softmax', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='ADModel')

    return model


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


def train_model():
    N = len(andi_data[0])
    X_train, Y_train = np.array(andi_data[0][:N]), np.array(
        andi_data[1][:N])

    """
        Set planes which have useful info
        from 7th to 73th
    """

    X_train = check_for_nans(X_train)
    ad_model = model(X_train[0].shape)
    ad_model.compile(optimizer='adam', loss='mean_absolute_error',
                     metrics=['accuracy'])
    ad_model.fit(x=X_train, y=Y_train, epochs=300, batch_size=64)


# create_dataset(filename='/Volumes/ELEMENT/Alzheimer/generated.h5')
# print('Finished!')
adni_data = h5.File('/Volumes/ELEMENT/Alzheimer/generated.h5', 'r')
reslice_volume(adni_data['features'], adni_data['labels'], [7, 74],
               '/Volumes/ELEMENT/Alzheimer/zeromin_re_7_74_generated.h5')

# path to saved dataset /Volumes/ELEMENT/Alzheimer/generated.pickle

# andi_data = ut.read_from_fld('/Volumes/ELEMENT/Alzheimer/generated.pickle')

# print(f'Reading time is: {start - time.time()}')
# reslice_volume(andi_data, [7, 73])
# # # exit()
# """
#  After reading format is a list:
#  `[0]` - images
#  `[1]` - labels
# """
# image = andi_data[0][0]
# ut.dump_to_fld('image.pickle', image)

# print('Saved!')
# train_model()
