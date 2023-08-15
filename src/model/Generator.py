import os
import pickle
import numpy as np
import tensorflow as tf

OUTPUT_CHANNELS = 3
# Define the downsample function for the generator model
def downsample(filters, size, apply_batchnorm=True):
    # Weight initialization
    initializer = initializer = tf.random_normal_initializer(0., 0.02)

    # Define the model
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            filters, size, strides=2, padding='same',
            kernel_initializer=initializer, use_bias=False
        )
    )

    # Apply batch normalization if required and add leaky ReLU activation function to the model
    if apply_batchnorm:
        model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.LeakyReLU())

    # Return the model
    return model


# Define the upsample function for the generator model
def upsample(filters, size, apply_dropout=False):
    # Weight initialization
    initializer = initializer = tf.random_normal_initializer(0., 0.02)

    # Define the model
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2DTranspose(
            filters, size, strides=2, padding='same',
            kernel_initializer=initializer, use_bias=False
        )
    )

    # Apply batch normalization if required and add ReLU activation function to the model
    model.add(tf.keras.layers.BatchNormalization())

    # Apply dropout if required
    if apply_dropout:
        model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.ReLU())

    # Return the model
    return model


def define_generator(in_shape=(256, 256, 3)):
    # Weight initialization
    inputs = tf.keras.layers.Input(shape=in_shape)

    # Downsampling through the model (encoder) - 8 layers of downsampling (2^8 = 256)
    down_stack = [
        downsample(64, 5, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 3),  # (batch_size, 64, 64, 128)
        downsample(128, 3),  # (batch_size, 32, 32, 256)
        downsample(256, 3),  # (batch_size, 16, 16, 512)
        downsample(256, 3),  # (batch_size, 8, 8, 512)
        downsample(256, 3),  # (batch_size, 4, 4, 512)
        downsample(256, 3),  # (batch_size, 2, 2, 512)
        downsample(256, 3),  # (batch_size, 1, 1, 512)
    ]

    # Upsampling through the model (decoder) - 8 layers of upsampling (2^8 = 256)
    up_stack = [
        upsample(256, 3, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(256, 3, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(256, 3, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(256, 3),  # (batch_size, 16, 16, 1024)
        upsample(128, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 3),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    # Weight initialization for the last layer of the model (output layer) - 3 channels for RGB image output
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    # Define the last layer of the model (output layer)
    # - 3 channels for RGB image output
    # - tanh activation function to ensure output is between -1 and 1
    last = tf.keras.layers.Conv2DTranspose(
        OUTPUT_CHANNELS, 4, strides=2, padding='same',
        kernel_initializer=initializer, activation='tanh'  # (batch_size, 256, 256, 3)
    )

    # Define the model
    x = inputs

    # Downsampling through the model (encoder) - 8 layers of downsampling (2^8 = 256)
    skips = []
    # Loop through the down_stack layers and apply the model to the input
    for down in down_stack:
        x = down(x)
        skips.append(x)

    # Reverse the order of the skips list to allow for the skip connections to be made in the upsampling layers
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        # Concatenate the skip connection with the upsampled output from the previous layer
        x = tf.keras.layers.Concatenate()([x, skip])

    # Apply the last layer of the model (output layer) to the output of the final upsampling layer
    x = last(x)

    # Return the model
    model = tf.keras.Model(inputs=inputs, outputs=x, name='Conditional_GAN')
    model.summary()

    return model