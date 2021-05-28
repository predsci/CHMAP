"""
Author: Tamar Ervin
Date: February 25, 2021

W-Net model for unsupervised image segmentation
goal: unsupervised method to detect coronal holes from EUV maps
- tracking over time??

W-Net Architecture: U-Net encoder to U-net decoder
"""

# imports
import h5py as h5
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import coronal_holes.ml_detect.ml_functions as ml_funs

# Image size that we are going to use
IMG_HEIGHT = 96
IMG_WIDTH = 240
# Our images are RGB (3 channels)
N_CHANNELS = 3
# Number of classes
N_CLASSES = 2
# Buffer and Batch Size
BUFFER_SIZE = 16
BATCH_SIZE = 32

# data
# read hdf5 file
map_h5 = '/Volumes/CHD_DB/map_data.h5'
hf = h5.File(map_h5, 'r')

# create list of images and masks
train_image = []
train_mask = []
for date in hf.keys():
    g = hf.get(date)
    train_image.append(np.array(g['euv_image']))
    train_mask.append(np.array(g['chd_data']))
hf.close()

### create tensorflow dataset
# convert to dictionaries
train_dict = {key: value for key, value in zip(["image"], [train_image])}
train_dict['segmentation_mask'] = train_mask

del train_image
del train_mask

map_norm = ml_funs.load_image_train(train_dict, size=(IMG_HEIGHT, IMG_WIDTH))
map_dset = tf.data.Dataset.from_tensor_slices(map_norm)
# train_dset = map_dset.map(ml_funs.load_image_train(size=(IMG_HEIGHT, IMG_WIDTH)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = map_dset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# convolution function
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
               padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    # should this be the input tensor??
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def sep_conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = SeparableConv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
               padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    # should this be the input tensor??
    x = Activation('relu')(x)

    # second layer
    x = SeparableConv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

# unet encoder model
def uenc(img, n_filters, kernel_size=3, dropout=0.1, batchnorm=True):
    # Contracting Path
    c1 = conv2d_block(img, n_filters * 1, kernel_size=kernel_size, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = sep_conv2d_block(p1, n_filters * 2, kernel_size=kernel_size, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = sep_conv2d_block(p2, n_filters * 4, kernel_size=kernel_size, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = sep_conv2d_block(p3, n_filters * 8, kernel_size=kernel_size, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = sep_conv2d_block(p4, n_filters=n_filters * 16, kernel_size=kernel_size, batchnorm=batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = sep_conv2d_block(u6, n_filters * 8, kernel_size=kernel_size, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = sep_conv2d_block(u7, n_filters * 4, kernel_size=kernel_size, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = sep_conv2d_block(u8, n_filters * 2, kernel_size=kernel_size, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=kernel_size, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[img], outputs=[outputs])
    return model


# unet decoder model
def udec(img, n_filters, kernel_size=3, dropout=0.1, batchnorm=True):
    # Softmax Layer
    sm = Softmax()
    c0 = sm(img)

    # Contracting Path
    c1 = conv2d_block(c0, n_filters * 1, kernel_size=kernel_size, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = sep_conv2d_block(p1, n_filters * 2, kernel_size=kernel_size, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = sep_conv2d_block(p2, n_filters * 4, kernel_size=kernel_size, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = sep_conv2d_block(p3, n_filters * 8, kernel_size=kernel_size, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = sep_conv2d_block(p4, n_filters=n_filters * 16, kernel_size=kernel_size, batchnorm=batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = sep_conv2d_block(u6, n_filters * 8, kernel_size=kernel_size, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = sep_conv2d_block(u7, n_filters * 4, kernel_size=kernel_size, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = sep_conv2d_block(u8, n_filters * 2, kernel_size=kernel_size, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=kernel_size, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[img], outputs=[outputs])
    return model

# soft normalized cut loss
def softncut_loss(V, K):
    J = None
    return J

# reconstruction loss
def recon_loss(V, K):
    R = None
    return R


print('la-di-da')
# compile models
input_image = Input((IMG_HEIGHT, IMG_WIDTH, N_CHANNELS), name='img')

uenc = uenc(input_image, n_filters=64, kernel_size=3, dropout=0.05, batchnorm=True)
uenc.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=["accuracy"])

udec = udec(input_image, n_filters=64, kernel_size=3, dropout=0.05, batchnorm=True)
udec.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=["accuracy"])


# minibatch gradient descent training loop
def wnet(x, uenc, udec, epochs):
    """Wnet convolutional model for unsupervised segmentation
    Parameters
    ----------
    x
    uenc
    udec
    epochs

    """

    # for i in epochs:
        # sample minibatch of new input images

        # update Uenc by minimizing Jsoft-Ncut

        # update W-net by minimizing Jreconstr

    return uenc


