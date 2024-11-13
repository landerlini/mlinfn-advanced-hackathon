import pytest

###########################################################################

def test_tensorflow():
    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    assert len(devices) > 0

    rnd = tf.random.uniform(shape=(100, 1))
    assert tf.math.abs(tf.reduce_sum(rnd)) > 0.0


def test_all_imports():
    # sMRI_fMRI_sep
    import os
    import logging
    import warnings

    import numpy as np
    import pandas as pd
    import shap

    import sklearn
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    from scipy import interp

    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.layers import BatchNormalization, Dropout
    from tensorflow.keras.backend import clear_session
    from tensorflow.keras.regularizers import l1
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

    import matplotlib.pyplot as plt

    # Joint_Fusion
    import os
    import time
    import logging
    import warnings

    import numpy as np
    import pandas as pd
    import shap

    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, BatchNormalization, Concatenate, Dropout
    from tensorflow.keras.regularizers import l1
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.backend import clear_session
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

    import sklearn
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve, auc

    import matplotlib.pyplot as plt
    from matplotlib import colormaps as cm
    from matplotlib.lines import Line2D

    from nilearn import datasets, plotting
