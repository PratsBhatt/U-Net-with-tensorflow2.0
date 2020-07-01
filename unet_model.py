import tensorflow as tf
from tensorflow.keras import layers


def unet_model():
    # declaring the input layer
    # Input layer expects an RGB image, in the original paper the network consisted of only one channel.
    inputs = layers.Input(shape=(572, 572, 3))
    # first part of the U - contracting part
    c0 = layers.Conv2D(64, activation='relu', kernel_size=3)(inputs)
    c1 = layers.Conv2D(64, activation='relu', kernel_size=3)(c0)  # This layer for concatenating in the expansive part
    c2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(c1)

    c3 = layers.Conv2D(128, activation='relu', kernel_size=3)(c2)
    c4 = layers.Conv2D(128, activation='relu', kernel_size=3)(c3)  # This layer for concatenating in the expansive part
    c5 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(c4)

    c6 = layers.Conv2D(256, activation='relu', kernel_size=3)(c5)
    c7 = layers.Conv2D(256, activation='relu', kernel_size=3)(c6)  # This layer for concatenating in the expansive part
    c8 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(c7)

    c9 = layers.Conv2D(512, activation='relu', kernel_size=3)(c8)
    c10 = layers.Conv2D(512, activation='relu', kernel_size=3)(c9)  # This layer for concatenating in the expansive part
    c11 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(c10)

    c12 = layers.Conv2D(1024, activation='relu', kernel_size=3)(c11)
    c13 = layers.Conv2D(1024, activation='relu', kernel_size=3, padding='valid')(c12)

    # We will now start the second part of the U - expansive part
    t01 = layers.Conv2DTranspose(512, kernel_size=2, strides=(2, 2), activation='relu')(c13)
    crop01 = layers.Cropping2D(cropping=(4, 4))(c10)

    concat01 = layers.concatenate([t01, crop01], axis=-1)

    c14 = layers.Conv2D(512, activation='relu', kernel_size=3)(concat01)
    c15 = layers.Conv2D(512, activation='relu', kernel_size=3)(c14)

    t02 = layers.Conv2DTranspose(256, kernel_size=2, strides=(2, 2), activation='relu')(c15)
    crop02 = layers.Cropping2D(cropping=(16, 16))(c7)

    concat02 = layers.concatenate([t02, crop02], axis=-1)

    c16 = layers.Conv2D(256, activation='relu', kernel_size=3)(concat02)
    c17 = layers.Conv2D(256, activation='relu', kernel_size=3)(c16)

    t03 = layers.Conv2DTranspose(128, kernel_size=2, strides=(2, 2), activation='relu')(c17)
    crop03 = layers.Cropping2D(cropping=(40, 40))(c4)

    concat03 = layers.concatenate([t03, crop03], axis=-1)

    c18 = layers.Conv2D(128, activation='relu', kernel_size=3)(concat03)
    c19 = layers.Conv2D(128, activation='relu', kernel_size=3)(c18)

    t04 = layers.Conv2DTranspose(64, kernel_size=2, strides=(2, 2), activation='relu')(c19)
    crop04 = layers.Cropping2D(cropping=(88, 88))(c1)

    concat04 = layers.concatenate([t04, crop04], axis=-1)

    c20 = layers.Conv2D(64, activation='relu', kernel_size=3)(concat04)
    c21 = layers.Conv2D(64, activation='relu', kernel_size=3)(c20)

    # This is based on our dataset. The output channels are 3, think of it as each pixel will be classified
    # into three classes, but I have written 4 here, as I do padding with 0, so we end up have four classes.
    outputs = layers.Conv2D(4, kernel_size=1)(c21)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="u-netmodel")
    return model
