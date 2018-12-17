import tensorflow as tf


def convolution_layer(inputs, filters, kernel_size, strides, gain=1.0):
    return tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=(strides, strides),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.orthogonal_initializer(gain=gain))


def fully_connected_layer(inputs, units, activation_fn, gain=1.0):
    return tf.layers.dense(inputs=inputs,
                           units=units,
                           activation=activation_fn,
                           kernel_initializer=tf.orthogonal_initializer(gain))
