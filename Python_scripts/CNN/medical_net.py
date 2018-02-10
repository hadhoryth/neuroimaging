from keras.applications.inception_resnet_v2 import preprocess_input, conv2d_bn, _obtain_input_shape, inception_resnet_block, get_source_inputs
from keras.layers import Input, MaxPooling2D, AveragePooling2D, Concatenate, GlobalMaxPooling2D, Dense
from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import h5py as h5
from sklearn.model_selection import train_test_split

RES_WEIGTHS_PATH = '/Volumes/ELEMENT/Alzheimer/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
MY_WEITGHS_PATH = '/Volumes/ELEMENT/Alzheimer/weights/'


def StemModule(prev_layer):
    stem = conv2d_bn(prev_layer, 72, 3, strides=2, padding='valid')
    stem = conv2d_bn(stem, 72, 3, padding='valid')
    stem = conv2d_bn(stem, 84, 3)
    stem = MaxPooling2D(3, strides=1)(stem)
    stem = conv2d_bn(stem, 90, 1, padding='valid')
    stem = conv2d_bn(stem, 192, 3, padding='valid')
    return MaxPooling2D(3, strides=2)(stem)


def create_hadnetV1(input_shape, useWeights=False):
    _input = Input(shape=input_shape)
    x = StemModule(_input)
    print(x.shape)
    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96, 1)
    branch_1 = conv2d_bn(x, 48, 1)
    branch_1 = conv2d_bn(branch_1, 64, 5)
    branch_2 = conv2d_bn(x, 64, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    x = Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='valid')
    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='valid')
    branch_2 = conv2d_bn(x, 256, 1)
    branch_2 = conv2d_bn(branch_2, 288, 3)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='valid')
    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x,
                                   scale=0.2,
                                   block_type='block8',
                                   block_idx=block_idx)
    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10)

    # Final convolution block: 3 x 4 x 1536
    x = conv2d_bn(x, 1536, 1, name='conv_7b')
    x = GlobalMaxPooling2D()(x)

    model = Model(get_source_inputs(_input), x, name='inception_hadnet_v2')
    if useWeights:
        model.load_weights(RES_WEIGTHS_PATH)
    return model


def read_data(file_path):
    data = h5.File(file_path, 'r')
    X_train, X_test, y_train, y_test = train_test_split(
        data['features'][:], data['labels'][:], test_size=0.1, random_state=42)
    data.close()
    return X_train, y_train


def train():
    features, labels = read_data('/Volumes/ELEMENT/Alzheimer/re_7_73_generated.h5')

    base_model = create_hadnetV1(input_shape=(79, 95, 66))
    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    x = Dense(300, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(MY_WEITGHS_PATH + 'weights-improvement-01-0.50.hdf5')

    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer='adam', loss='mean_absolute_error',
                  metrics=['accuracy'])
    filepath = MY_WEITGHS_PATH + 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')
    model.fit(x=features, y=labels, epochs=100, batch_size=64,
              callbacks=[checkpoint], validation_split=0.2)


if __name__ == '__main__':
    train()
