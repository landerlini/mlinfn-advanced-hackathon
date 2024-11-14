import pytest

###########################################################################

def test_tensorflow():
    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    assert len(devices) > 0

    rnd = tf.random.uniform(shape=(100, 1))
    assert tf.math.abs(tf.reduce_sum(rnd)) > 0.0


def test_all_imports():
    import os
    import sys
    import time
    import operator
    from tqdm import tqdm

    import numpy as np

    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, Concatenate, Add
    from tensorflow.keras.layers import Layer, BatchNormalization, Activation, ZeroPadding2D, LeakyReLU
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import backend as K

    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, KernelCenterer, RobustScaler, MaxAbsScaler, MinMaxScaler

    import pylab as pyy

    import matplotlib.pyplot as plt
    import matplotlib.cm, matplotlib.colors
    from matplotlib.legend_handler import HandlerLine2D
    from matplotlib.colors import LogNorm
