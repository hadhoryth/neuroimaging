import h5py as h5
import os
from sklearn.model_selection import train_test_split
from keras.layers import Activation, AveragePooling3D, BatchNormalization, Concatenate, Conv3D
from keras.layers import GlobalMaxPooling3D, Dense, GlobalMaxPool3D, Input, Lambda
from keras.layers import MaxPooling3D, Flatten
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import ModelCheckpoint


DATA_PATH = 'dataset/equalized_by_ad.h5'


def get_train_data(filepath=DATA_PATH):
    data = h5.File(filepath, 'r')
    X_train, X_test, y_train, y_test = train_test_split(
        data['features'][:], data['labels'][:], test_size=0.1, random_state=42)
    data.close()
    return X_train, y_train


def clear_output():
    f_name = 'simple_network.output.txt'
    try:
        os.remove('Inception_ResNet_3D.output.txt')
        print('File {} was deleted'.format(f_name))
    except OSError:
        print('File {} does not exist!'.format(f_name))

# TODO Turn it to the ALEXNET ---> VGG ---> GoogleNET


def model(_x, _y):
    _inputs = Input(shape=_x.shape[1:])
    x = Conv3D(filters=32, kernel_size=3)(_inputs)
    x = Conv3D(filters=64, kernel_size=3, strides=2)(x)
    x = MaxPooling3D()(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=128, kernel_size=1)(x)
    x = Conv3D(filters=192, kernel_size=3, strides=2)(x)
    x = MaxPooling3D()(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(3, activation='softmax')(x)
    return Model(_inputs, x)


if __name__ == '__main__':
    clear_output()
    print('Loading features .................. ', end='')
    features, labels = get_train_data()
    print('OK!')

    print('Creating a model .................. ', end='')
    model = model(features, labels)
    # base_model.summary(line_length=120)
    print('OK!')

    print('Compiling the model .................. ', end='')
    opt = Adam(lr=0.00001, decay=0.001)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
    print('OK!')

    print('Start training')
    model.fit(x=features, y=labels, batch_size=32, epochs=500)
