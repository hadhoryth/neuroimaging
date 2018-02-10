
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import h5py as h5
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Activation, AveragePooling3D, BatchNormalization, Concatenate, Conv3D
from keras.layers import GlobalMaxPooling3D, Dense, GlobalMaxPool3D, Input, Lambda
from keras.layers import MaxPooling3D, Flatten
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from keras.utils import plot_model

# cloud: /home/javierrp/ivan/scripts
MY_WEITGHS_PATH = '/home/javierrp/ivan/scripts/weights/'


def conv3d_bn(x, filters, kernel_size, strides=1, padding='same', activation='relu',
              use_bias=False, name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv3D`.
        kernel_size: kernel size as in `Conv3D`.
        strides: strides in `Conv3D`.
        padding: padding mode in `Conv3D`.
        activation: activation in `Conv3D`.
        use_bias: whether to use a bias in `Conv3D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.

    # Returns
        Output tensor after applying `Conv3D` and `BatchNormalization`.
    """
    x = Conv3D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias,
               name=name)(x)

    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 4
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    if block_type == 'block35':
        branch_0 = conv3d_bn(x, 32, 1)
        branch_1 = conv3d_bn(x, 32, 1)
        branch_1 = conv3d_bn(branch_1, 32, 3)
        branch_2 = conv3d_bn(x, 32, 1)
        branch_2 = conv3d_bn(branch_2, 48, 3)
        branch_2 = conv3d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv3d_bn(x, 192, 1)
        branch_1 = conv3d_bn(x, 128, 1)
        branch_1 = conv3d_bn(branch_1, 160, [1, 7, 1])
        branch_1 = conv3d_bn(branch_1, 192, [7, 1, 1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv3d_bn(x, 192, 1)
        branch_1 = conv3d_bn(x, 192, 1)
        branch_1 = conv3d_bn(branch_1, 224, [1, 3, 1])
        branch_1 = conv3d_bn(branch_1, 256, [3, 1, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 4
    mixed = Concatenate(axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv3d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=block_name + '_conv')

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale},
               name=block_name)([x, up])
    if activation is not None:
        x = Activation(activation, name=block_name + '_ac')(x)
    return x


def stem_block(x):
    # Stem block: 16 x 20 x 13 x 192
    x = Conv3D(filters=32, kernel_size=3)(x)

    # x = Conv3D(filters=32, kernel_size=3)(x)
    x = Conv3D(filters=64, kernel_size=3, strides=2)(x)

    conv_low = Conv3D(filters=96, kernel_size=2)(x)
    conv_deep = Conv3D(filters=64, kernel_size=2)(x)
    x = Concatenate()([conv_low, conv_deep])
    x = BatchNormalization()(x)

    conv_1x1 = Conv3D(filters=64, kernel_size=1, padding='same')(x)
    branch_0 = Conv3D(filters=96, kernel_size=3)(conv_1x1)
    branch_1 = Conv3D(filters=64, kernel_size=(7, 1, 1), padding='same')(conv_1x1)
    branch_1 = Conv3D(filters=64, kernel_size=(1, 7, 1), padding='same')(branch_1)
    branch_1 = Conv3D(filters=128, kernel_size=3)(branch_1)
    x = Concatenate()([branch_0, branch_1])
    x = MaxPooling3D()(x)
    x = BatchNormalization()(x)
    return Activation(activation='relu')(x)


def InceptionResNetV2(input_shape=None, classes=3):
    _inputs = Input(shape=input_shape)
    x = stem_block(_inputs)

    # Mixed 5b (Inception-A block): 17x21x14x320
    branch_0 = conv3d_bn(x, 96, 1)
    branch_1 = conv3d_bn(x, 48, 1)
    branch_1 = conv3d_bn(branch_1, 64, 5)
    branch_2 = conv3d_bn(x, 64, 1)
    branch_2 = conv3d_bn(branch_2, 96, 3)
    branch_2 = conv3d_bn(branch_2, 96, 3)
    branch_pool = AveragePooling3D(3, strides=1, padding='same')(x)
    branch_pool = conv3d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(name='mixed_5b')(branches)
    # 10x block35 (Inception-ResNet-A block): 17x21x14x320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x, scale=0.17, block_type='block35', block_idx=block_idx)

    branch_0 = conv3d_bn(x, 384, 3, strides=2, padding='valid')
    branch_1 = conv3d_bn(x, 256, 1)
    branch_1 = conv3d_bn(branch_1, 256, 3)
    branch_1 = conv3d_bn(branch_1, 384, 3, strides=2, padding='valid')
    branch_pool = MaxPooling3D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 7 x 9 x 5 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x, scale=0.1, block_type='block17', block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 7 x 9 x 5 x 1088

    x = Concatenate(name='mixed_6a')(branches)

    # 10x block8 (Inception-ResNet-C block): 7 x 9 x 5 x 1088
    for block_idx in range(1, 10):
        x = inception_resnet_block(x, scale=0.2, block_type='block8', block_idx=block_idx)

    x = inception_resnet_block(x, scale=1., activation=None, block_type='block8', block_idx=10)

    # Final convolution block: 7 x 9 x 5 x 1536
    x = conv3d_bn(x, 1800, 1, name='conv_7b')
    x = GlobalMaxPooling3D()(x)
    # x = Flatten()(x)

    # Create model
    model = Model(_inputs, x, name='inception_resnet_v2_3d')

    return model


def read_data(file_path):
    data = h5.File(file_path, 'r')
    X_train, X_test, y_train, y_test = train_test_split(
        data['features'][:], data['labels'][:], test_size=0.1, random_state=42)
    data.close()
    return X_train, y_train


def clear_output():
    try:
        os.remove('Inception_ResNet_3D.output.txt')
        print('Output file was deleted')
    except OSError:
        print('Such file does not exist!')


if __name__ == '__main__':
    clear_output()
    width = 79
    height = 95
    depth = 67

    # cloud: /home/javierrp/ivan/dataset/
    features, labels = read_data('/home/javierrp/ivan/dataset/equalized_by_ad.h5')
    base_model = InceptionResNetV2(input_shape=(width, height, depth, 1))

    x = base_model.output
    x = Dense(700, activation='relu')(x)
    x = Dense(300, activation='relu')(x)
    prediction = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=prediction)

    # model.load_weights(MY_WEITGHS_PATH + 'weights-improvement-01-0.50.hdf5')
    opt = Adam(lr=0.01, decay=0.001)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
    # with open('report.txt', 'w') as fh:
    #    model.summary(print_fn=lambda x: fh.write(x + '\n'))
    # exit()

    filepath = MY_WEITGHS_PATH + 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')

    model.fit(x=features, y=labels, batch_size=16, epochs=500,
              callbacks=[checkpoint], validation_split=0.1, )
