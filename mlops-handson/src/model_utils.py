# used as dataloader
import numpy as np
import keras
from keras import layers
import tensorflow as tf 

from numpy.lib.recfunctions import structured_to_unstructured
from functools import partial
import random
import os

def set_global_seeds(seed=42):
    # Set seeds for all random number generators
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Enable TensorFlow determinism
    tf.config.experimental.enable_op_determinism()
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

### DATALOADER UTILS ################################################################
def load_data(point_cloud_batch, label_cloud_batch, n_points=800, n_features=3, labels=["unfocus hit", "focus hit"]):
    point_cloud_batch.set_shape([n_points, n_features])
    label_cloud_batch.set_shape([n_points, len(labels)])
    return point_cloud_batch, label_cloud_batch


def augment(point_cloud_batch, label_cloud_batch):
    noise = tf.random.uniform(
        tf.shape(point_cloud_batch[:, :, :3]), -0.001, 0.001, dtype=tf.float64
    )

    noisy_xyz = point_cloud_batch[:, :, :3] + noise
    point_cloud_batch = tf.concat([noisy_xyz, point_cloud_batch[:, :, 3:]], axis=-1)
    
    return point_cloud_batch, label_cloud_batch


def generate_dataset(point_clouds, label_clouds, is_training=True, bs=16, n_points=800, n_features=3, labels=["unfocus hit", "focus hit"]):
    # reformat to unstructured array and transform to list of size n_samples, each element of size n_ponints x n_features
    point_clouds = structured_to_unstructured(point_clouds).astype(np.float64)
    point_clouds = [_ for _ in point_clouds]
    
    dataset = tf.data.Dataset.from_tensor_slices((point_clouds, label_clouds))
    dataset = dataset.shuffle(bs * 100) if is_training else dataset
    load_data_with_args = partial(load_data, n_points=n_points, n_features=n_features, labels=labels)
    dataset = dataset.map(load_data_with_args, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size=bs)
    dataset = (
        dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        if is_training
        else dataset
    )
    return dataset

#####################################################################################

### MODEL UTILS #####################################################################
def conv_block(x, filters, name):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid", name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def mlp_block(x, filters, name):
    x = layers.Dense(filters, name=f"{name}_dense")(x)
    x = layers.BatchNormalization(name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    """Reference: https://keras.io/examples/vision/pointnet/#build-a-model"""

    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.identity = keras.ops.eye(num_features)

    def __call__(self, x):
        x = keras.ops.reshape(x, (-1, self.num_features, self.num_features))
        xxt = keras.ops.tensordot(x, x, axes=(2, 2))
        xxt = keras.ops.reshape(xxt, (-1, self.num_features, self.num_features))
        return keras.ops.sum(self.l2reg * keras.ops.square(xxt - self.identity))

    def get_config(self):
        config = super().get_config()
        config.update({"num_features": self.num_features, "l2reg_strength": self.l2reg})
        return config

def transformation_net(inputs, num_features, init_size, name):
    """
    Reference: https://keras.io/examples/vision/pointnet/#build-a-model.

    The `filters` values come from the original paper:
    https://arxiv.org/abs/1612.00593.
    """
    x = conv_block(inputs, filters=init_size, name=f"{name}_1")
    x = conv_block(x, filters=init_size*2, name=f"{name}_2")
    x = conv_block(x, filters=init_size*32, name=f"{name}_3")
    x = layers.GlobalMaxPooling1D()(x)
    x = mlp_block(x, filters=init_size*8, name=f"{name}_1_1")
    x = mlp_block(x, filters=init_size*4, name=f"{name}_2_1")
    return layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(np.eye(num_features).flatten()),
        activity_regularizer=OrthogonalRegularizer(num_features),
        name=f"{name}_final",
    )(x)


def transformation_block(inputs, num_features, init_size, name):
    transformed_features = transformation_net(inputs, num_features, init_size, name=name)
    transformed_features = layers.Reshape((num_features, num_features))(
        transformed_features
    )
    return layers.Dot(axes=(2, 1), name=f"{name}_mm")([inputs, transformed_features])

def get_shape_segmentation_model(num_points, num_classes, n_features, init_size, end_size):
    input_points = keras.Input(shape=(None, n_features))

    # PointNet Classification Network.
    transformed_inputs = transformation_block(
        input_points, num_features=n_features, init_size=init_size, name="input_transformation_block"
    )
    features_init = conv_block(transformed_inputs, filters=init_size, name=f"features_{init_size}")
    features_2x_1 = conv_block(features_init, filters=init_size*2, name=f"features_{init_size*2}_1")
    features_2x_2 = conv_block(features_2x_1, filters=init_size*2, name=f"features_{init_size*2}_2")
    transformed_features = transformation_block(
        features_2x_2, num_features=init_size*2, init_size=init_size, name="transformed_features"
    )
    features_8x = conv_block(transformed_features, filters=init_size*8, name=f"features_{init_size*8}")
    features_32x = conv_block(features_8x, filters=init_size*32, name="pre_maxpool_block")
    global_features = layers.MaxPool1D(pool_size=num_points, name="global_features")(
        features_32x
    )
    global_features = keras.ops.tile(global_features, [1, num_points, 1])

    # Segmentation head.
    segmentation_input = layers.Concatenate(name="segmentation_input")(
        [
            features_init,
            features_2x_1,
            features_2x_2,
            transformed_features,
            features_8x,
            global_features,
        ]
    )
    segmentation_features = conv_block(
        segmentation_input, filters=end_size, name="segmentation_features"
    )
    outputs = layers.Conv1D(
        num_classes, kernel_size=1, activation="softmax", name="segmentation_head"
    )(segmentation_features)
    return keras.Model(input_points, outputs)


#####################################################################################

### TRAINING UTILS ##################################################################
