import os
import pickle
import numpy as np
import tensorflow as tf

def define_discriminator(in_shape=(256, 256, 3)):
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    global_disc_input = tf.keras.layers.Input(shape=in_shape)

    x = tf.keras.layers.Conv2D(32, 5, padding='same', strides=(2, 2), kernel_initializer=init)(global_disc_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Conv2D(64, 5, padding='same', strides=(2, 2), kernel_initializer=init)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Conv2D(128, 5, padding='same', strides=(2, 2), kernel_initializer=init)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Conv2D(256, 5, padding='same', strides=(2, 2), kernel_initializer=init)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Conv2D(256, 5, padding='same', strides=(2, 2), kernel_initializer=init)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    global_disc_output = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=init)(x)

    model = tf.keras.models.Model(inputs=global_disc_input, outputs=global_disc_output, name='Global_Discriminator')
    model.summary()
    return model

def define_PGAN_discriminator(in_shape=(256, 256, 3)):
    # weight initialization
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    # source image input
    in_src_image = tf.keras.layers.Input(shape=in_shape)
    # C64
    d = tf.keras.layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_src_image)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    # C128
    d = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = tf.keras.layers.BatchNormalization()(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    # C256
    d = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = tf.keras.layers.BatchNormalization()(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    # C512
    d = tf.keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = tf.keras.layers.BatchNormalization()(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = tf.keras.layers.Conv2D(256, (4, 4), padding='same', kernel_initializer=init)(d)
    d = tf.keras.layers.BatchNormalization()(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    # patch output
    d = tf.keras.layers.Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = tf.keras.layers.Activation('sigmoid')(d)
    # define model
    model = tf.keras.models.Model(inputs=in_src_image, outputs=patch_out, name='PGAN_Discriminator')
    model.summary()
    return model
