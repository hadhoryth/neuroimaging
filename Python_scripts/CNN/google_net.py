"""
    Implementation from
    https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14
    NOT WORKING 
    NEEDS REVIEW
"""
from keras.layers.core import Layer
from keras.layers import Input, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, concatenate, BatchNormalization
from keras.regularizers import l2
import copy


class PoolHelper(Layer):

    def __init__(self, **kwargs):
        super(PoolHelper, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[:, :, 1:, 1:]

    def get_config(self):
        config = {}
        base_config = super(PoolHelper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def PoolingModule(k, prev_layer):
    conv_zero_pad = ZeroPadding2D(padding=(1, 1))(prev_layer)
    # pool_helper = PoolHelper()(conv_zero_pad)
    return MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid',
                        name=f'pool{k}/3x3_s2')(conv_zero_pad)


def StemModule2D(k, prev_layer):
    conv_7x7_s2 = Convolution2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same',
                                activation='relu', name=f'conv{k}/7x7_s2',
                                W_regularizer=l2(0.0002))(prev_layer)

    pool_3x3_s2 = PoolingModule(k, conv_7x7_s2)
    pool1_norm = BatchNormalization(axis=3, name=f'pool{k}/norm1')(pool_3x3_s2)
    k += 1
    conv_3x3_reduce = Convolution2D(64, (1, 1), padding='same', activation='relu',
                                    name=f'conv{k}/3x3_reduce', W_regularizer=l2(0.0002))(pool1_norm)
    conv_3x3 = Convolution2D(192, (3, 3), padding='same', activation='relu',
                             name=f'conv{k}/3x3', W_regularizer=l2(0.0002))(conv_3x3_reduce)
    conv_norm2 = BatchNormalization(axis=3, name=f'conv{k}/norm2')(conv_3x3)
    return PoolingModule(k, conv_norm2)


def Inception2D(k, prev_layer, filters, reg_value=0.002):

    inception_1x1 = Convolution2D(filters[0], kernel_size=(1, 1), padding='same', activation='relu', name=f'inception{k}/1x1', W_regularizer=l2(reg_value))(prev_layer)
    inception_3x3_reduce = Convolution2D(filters[1], kernel_size=(1, 1), padding='same', activation='relu', name=f'inception{k}/3x3_reduce', W_regularizer=l2(reg_value))(prev_layer)
    inception_3x3 = Convolution2D(filters[2], kernel_size=(3, 3), padding='same', activation='relu', name=f'inception{k}/3x3', W_regularizer=l2(reg_value))(inception_3x3_reduce)
    inception_5x5_reduce = Convolution2D(filters[3], kernel_size=(1, 1), padding='same', activation='relu', name=f'inception{k}/5x5_reduce', W_regularizer=l2(reg_value))(prev_layer)
    inception_5x5 = Convolution2D(filters[4], kernel_size=(5, 5), padding='same', activation='relu', name=f'inception{k}/5x5', W_regularizer=l2(reg_value))(inception_5x5_reduce)
    inception_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name=f'inception_{k}d/pool')(prev_layer)
    inception_pool_proj = Convolution2D(filters[5], kernel_size=(1, 1), padding='same', activation='relu', name=f'inception{k}/pool_proj', W_regularizer=l2(reg_value))(inception_5x5_reduce)

    return concatenate([inception_1x1, inception_3x3, inception_5x5, inception_pool_proj], name=f'inception_{k}d/output')


def InceptionModule2D(k, prev_layer, filters):
    applied_to = prev_layer
    for kk, filts in enumerate(filters):
        inception_out = Inception2D(str(k), applied_to, filts)
        applied_to = copy.copy(inception_out)
    return applied_to


def create_googlenet(input_shape, weights_path=None):
    _input = Input(shape=input_shape)
    # Stem
    stem_out = StemModule2D(1, _input)

    # Inception modules
    filters_3 = [[64, 96, 128, 16, 32, 32], [128, 128, 192, 32, 96, 64]]
    inception_3_output = InceptionModule2D(3, stem_out, filters_3)
    inception_block3_output = PoolingModule(3, inception_3_output)

    filters_4 = [[192, 96, 208, 16, 48, 64], [160, 112, 224, 24, 64, 64], [112, 128, 256, 24, 64, 64],
                 [112, 114, 288, 32, 64, 64], [256, 160, 320, 32, 128, 128]]
    inception_4_output = InceptionModule2D(4, inception_block3_output, filters_4)
    inception_block4_output = PoolingModule(3, inception_4_output)

    filters_5 = [[256, 160, 320, 32, 128, 128], [384, 192, 384, 48, 128, 128]]
    inception_5_output = InceptionModule2D(5, inception_block4_output, filters_5)

    # Out Classifier
    pool5_7x7_s1 = AveragePooling2D(pool_size=(7, 7), strides=(
        1, 1), name='pool5/7x7_s2')(inception_5_output)

    loss3_flat = Flatten()(pool5_7x7_s1)
    pool5_drop_7x7_s1 = Dropout(0.4)(loss3_flat)
    loss3_classifier = Dense(3, name='loss3/classifier',
                             W_regularizer=l2(0.0002))(pool5_drop_7x7_s1)
    loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)

    googlenet = Model(input=_input, output=loss3_classifier)

    if weights_path:
        googlenet.load_weigths(weights_path)

    return googlenet


model = create_googlenet(input_shape=(3, 224, 224))
print('Model created!')
