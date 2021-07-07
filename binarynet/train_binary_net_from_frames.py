#https://github.com/fastmachinelearning/keras-training

from __future__ import print_function
import numpy as np

import keras.backend as K
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
from keras import regularizers
from tensorflow.keras.models import load_model
from binarynet.binary_ops import binary_tanh as binary_tanh_op
from binarynet.binary_layers import BinaryDense, BinaryConv2D

import cv2
import glob

def binary_tanh(x):
    return binary_tanh_op(x)

def train_binary_net(img_path, model_name, resolution, dense_layer_size, debug=False):
    file_list = glob.glob(img_path)

    H = 1.
    kernel_lr_multiplier = 'Glorot'

    # nn
    batch_size = 16
    epochs = 1000
    channels = 3
    img_rows = img_cols = resolution
    filters = 32
    kernel_size = (3, 3)
    pool_size = (2, 2)
    classes = 100
    use_bias = False

    # learning rate schedule
    lr_start = 1e-3
    lr_end = 1e-4
    lr_decay = (lr_end / lr_start)**(1. / epochs)

    # BN
    epsilon = 1e-6
    momentum = 0.9

    nb_samples = classes

    X_train = np.zeros((nb_samples, img_rows, img_cols, 3))
    y_train = np.zeros((nb_samples))

    for rep in range(1):
        K.clear_session()

        np.random.shuffle(file_list)

        for i, frame in enumerate(file_list):
            if i == nb_samples:
                break

            img = cv2.imread(frame)
            img = cv2.resize(img, (img_rows, img_cols))

            X_train[i, :, :, :] = img
            #y_train[i] = np.random.randint(classes)
            y_train[i] = i


        X_train = X_train.astype('float32')
        X_train /= 255

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, classes) * 2 - 1 # -1 or 1 for hinge loss


        model = Sequential()
        # conv1
        model.add(BinaryConv2D(filters, kernel_size=kernel_size, input_shape=(img_rows, img_cols, channels),
                               data_format='channels_last',
                               H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                               padding='same', use_bias=use_bias, name='conv1'))
        model.add(MaxPooling2D(pool_size=pool_size, data_format='channels_last'))
        model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn1'))
        model.add(Activation(binary_tanh, name='act1'))

        # model.add(BinaryConv2D(filters, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
        #                        data_format='channels_last',
        #                        padding='same', use_bias=use_bias))
        # model.add(MaxPooling2D(pool_size=pool_size, data_format='channels_last'))
        # model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1))
        # model.add(Activation(binary_tanh))

        model.add(BinaryConv2D(filters, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                               data_format='channels_last',
                               padding='same', use_bias=use_bias))
        model.add(MaxPooling2D(pool_size=pool_size, data_format='channels_last'))
        model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1))
        model.add(Activation(binary_tanh))

        model.add(Flatten())

        # dense1
        model.add(BinaryDense(dense_layer_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense5'))
        model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn5'))
        #model.add(Activation(binary_tanh, name='act5', activity_regularizer=regularizers.l1(0.001)))
        #model.add(Activation(binary_tanh, name='act5', activity_regularizer=regularizers.l1(0.00001)))
        model.add(Activation(binary_tanh, name='act5'))

        # dense2
        model.add(BinaryDense(classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense6'))
        model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn6'))

        opt = Adam(lr=lr_start)
        model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])

        print()
        print('Summary of the binary model:')
        model.summary()
        print()

        #input()

        if rep > 0:
            model = load_model(model_name)

        lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size, epochs=epochs,
                            verbose=1,
                            callbacks=[lr_scheduler])

        # summarize history for accuracy
        import matplotlib.pyplot as plt
        plt.plot(history.history['acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        #plt.savefig(str(filters)+'_2_conv_reg.png')
        if debug:
            plt.show()

        #model.save('Linnaeus_5_64X64_model')
        #model.save('Random_64X64_model')
        #model.save('Reacher_128X128_model')
        #model.save('Pong_80x80_model')
        model.save(model_name)

        #model.save('Linnaeus_5_64X64_model')
        #model.save('Linnaeus_5_128X128_model')

if __name__ == "__main__":
    train_img_path =  'Img_DB/Linnaeus 5 256x256/other/*.jpg'
    resolution = 64
    dense_layer_size = 128

    binary_model_name = 'Linnaeus_5_'+str(resolution)+'x'+str(resolution)+'_'+str(dense_layer_size)+'_model'

    train_binary_net(img_path=train_img_path, model_name=binary_model_name, resolution=resolution,
                     dense_layer_size=dense_layer_size, debug=True)
