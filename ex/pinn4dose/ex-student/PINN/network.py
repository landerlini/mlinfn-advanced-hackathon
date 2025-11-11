from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import GlorotNormal
import tensorflow as tf

def NN(size_input, size_output, sequence_of_layers=[10,20,200,20,10], activation="tanh", in_seed=100):
    initializer=GlorotNormal(in_seed)

    inputs=Input(shape=(size_input,), dtype=tf.float64)
    i=0
    for units in sequence_of_layers:
        if i==0:
            z=Dense(units,activation=activation, kernel_initializer=initializer)(inputs)
        else:
            z=Dense(units,activation=activation, kernel_initializer=initializer)(z)
        i+=1
    
    outputs=Dense(size_output, kernel_initializer=initializer)(z)

    model=tf.keras.Model(inputs,outputs)
    return model

