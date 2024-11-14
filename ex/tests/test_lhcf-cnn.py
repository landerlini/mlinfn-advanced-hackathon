import pytest

###########################################################################

def test_tensorflow():
    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    assert len(devices) > 0

    rnd = tf.random.uniform(shape=(100, 1))
    assert tf.math.abs(tf.reduce_sum(rnd)) > 0.0


def test_all_imports():
    # train_and_split
    import os
    import h5py
    import tensorflow as tf
    import numpy as np
    from sklearn.model_selection import train_test_split
    from multiprocessing import Pool, Manager
    from tqdm import tqdm

    # Network
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Conv2D, Conv1D, Dense,  MaxPooling2D, MaxPooling1D, Flatten, Concatenate, GlobalAveragePooling2D, GlobalAveragePooling1D
    from tensorflow.keras.models import Model
    from tensorflow import keras
    import matplotlib.pyplot as plt
    import numpy as np
    import time
