
"""
testing some ideas about GANs
"""

import h5py as h5
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import analysis.ml_analysis.ml_functions as ml_funs

from IPython import display

# load training data
BUFFER_SIZE = 16
BATCH_SIZE = 32

train_h5 = 'analysis/ml_analysis/data_train.h5'

hf = h5.File(train_h5, 'r')
train_image = []
train_mask = []
for date in hf.keys():
    g = hf.get(date)
    train_image.append(np.array(g['euv_image']))
    train_mask.append(np.array(g['chd_data']))

train_dict = {key: value for key, value in zip(["image"], [train_image])}
train_dict['segmentation_mask'] = train_mask

train_dataset = tf.data.Dataset.from_tensor_slices(train_dict).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train = train_dataset.map(ml_funs.load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
TRAIN_LENGTH = len(train_dataset)
train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# make generator model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


# discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# cross entropy loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# discriminator loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# generator loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


