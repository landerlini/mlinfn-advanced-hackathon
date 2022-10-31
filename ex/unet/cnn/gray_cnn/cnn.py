import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, SpatialDropout2D, Dense, Lambda, Conv2D, Concatenate, Reshape, MaxPooling2D, Activation, Flatten, Dropout, LeakyReLU, GlobalAvgPool2D, Add, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.activations import softmax
#from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


# definition of an advanced activation function
Optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
alpha = 0.2

def cnn_model(input_size):
# model generation
    input_cnn= Input(input_size)

    x = Conv2D(16, (3,3), padding="same")(input_cnn)

    for i in range(2):
        x = Conv2D(16, (3,3), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha)(x)
        x = BatchNormalization()(x)
        x = SpatialDropout2D(0.1)(x)
        x = MaxPooling2D(pool_size=(2,2))(x)

    for i in range(2):
        x = Conv2D(32, (3,3), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha)(x)
        x = BatchNormalization()(x)
        x = SpatialDropout2D(0.1)(x)
        x = MaxPooling2D(pool_size=(2,2))(x)


    x = GlobalAvgPool2D()(x)
    x = Dropout(0.1)(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)

    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=input_cnn, outputs=x)
    return model

