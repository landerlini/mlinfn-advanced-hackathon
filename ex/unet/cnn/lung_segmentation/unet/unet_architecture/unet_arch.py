# Copyright (C) 2021 A. Retico for the AIM-COVID19-WG of developers.
#
# This file is part of Analysis_PIPELINE_LungQuant.
#
# upsilon_analysis is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.



import numpy as np
import os
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import regularizers, initializers, constraints
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Input, BatchNormalization, SpatialDropout2D, Conv2D, Conv2DTranspose, Dropout, Concatenate, Lambda, Activation, Reshape, Add
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf


# In[41]:


def unet_layer_left(previous_layer, n_filters, ker_size_1, strides_1, ker_size_2, strides_2, ker_size_3, strides_3, reg):
    '''Definition of layers of the left part of the u-net. 
    Parameters
    ------------------
    previous_layer : input layer for this block;
    n_filters : number of filters of all the 3D convolutional layers;
    ker_size_1 : kernel dimension of the first conv3D layer of this block;
    strides_1 : strides size to be used for the convolution of the first Conv3D layer;
    ker_size_2/3 : kernel dimension of the second/third Conv3D layers;
    strides_2/3 : strides size to be used for the convolution of the second/third Conv3D layer;
    '''
    layer_L=Conv2D(filters=n_filters,kernel_size=ker_size_1,strides=strides_1,kernel_regularizer=regularizers.l2(reg), kernel_initializer='random_normal')(previous_layer)
    layer_L=BatchNormalization(axis=-1)(layer_L)  
    layer_L_shortcut=layer_L # this shortcut is needed to make the residual block
    layer_L=Activation('relu')(layer_L)
    layer_L=SpatialDropout2D(0.2)(layer_L)
    layer_L=Conv2D(filters=n_filters,kernel_size=ker_size_2,strides=strides_2,padding='same',kernel_regularizer=regularizers.l2(reg), kernel_initializer='random_normal')(layer_L)
    layer_L=BatchNormalization(axis=-1)(layer_L)  
    layer_L=Activation('relu')(layer_L)
    layer_L=SpatialDropout2D(0.2)(layer_L)
    layer_L=Conv2D(filters=n_filters,kernel_size=ker_size_3,strides=strides_3,padding='same',kernel_regularizer=regularizers.l2(reg), kernel_initializer='random_normal')(layer_L)
    layer_L=BatchNormalization(axis=-1)(layer_L)  
    layer_L=Add()([layer_L,layer_L_shortcut])
    layer_L=Activation('relu')(layer_L)
    layer_L=SpatialDropout2D(0.2)(layer_L)
    return layer_L


# In[40]:


def unet_layer_bottleneck(previous_layer, n_filters, ker_size_1, strides_1, ker_size_2, strides_2, reg):
    '''Definition of the last layer of the network. 
    Parameters
    ------------------
    previous_layer : input layer for this block;
    n_filters : number of filters of all the 3D convolutional layers;
    ker_size_1 : kernel dimension of the first conv3D layer of this block;
    strides_1 : strides size to be used for the convolution of the first Conv3D layer;
    ker_size_2 : kernel dimension of the second Conv3D layers;
    strides_2 : strides size to be used for the convolution of the second Conv3D layer;
    '''
    layer_L=Conv2D(filters=n_filters,kernel_size=ker_size_1,strides=strides_1,kernel_regularizer=regularizers.l2(reg), kernel_initializer='random_normal')(previous_layer)
    layer_L=BatchNormalization(axis=-1)(layer_L)  
    layer_L_shortcut=layer_L # this shortcut is needed to make the residual block
    layer_L=Activation('relu')(layer_L)
    layer_L=Conv2D(filters=n_filters,kernel_size=ker_size_2,strides=strides_2,padding='same',kernel_regularizer=regularizers.l2(reg), kernel_initializer='random_normal')(layer_L)
    layer_L=BatchNormalization(axis=-1)(layer_L)    
    layer_L=Add()([layer_L,layer_L_shortcut])
    layer_L=Activation('relu')(layer_L)
    return layer_L


# In[39]:


def unet_layer_right(previous_layer, layer_left, n_filters, ker_size_1, strides_1, output_pad, ker_size_2, strides_2, ker_size_3, strides_3, reg):
    '''Definition of layers of the right part of the u-net. 
    Parameters
    ------------------
    previous_layer : input layer for this block;
    layer_left : output layer of the left part at the same depth;
    n_filters : int, number of filters of all the 3D convolutional layers;
    ker_size_1 : int, kernel dimension of the first conv3D layer of this block;
    strides_1 : int, strides size to be used for the convolution of the first Conv3D layer;
    output_pad : tuple, padding for the output;
    ker_size_2/3 : int, kernel dimension of the second/third Conv3D layers;
    strides_2/3 : int, strides size to be used for the convolution of the second/third Conv3D layer;
    '''
    layer_R=Conv2DTranspose(filters=n_filters,kernel_size=ker_size_1,strides=strides_1,output_padding=output_pad,kernel_regularizer=regularizers.l2(reg), kernel_initializer='random_normal')(previous_layer)
    layer_R=BatchNormalization(axis=-1)(layer_R)
    layer_R_shortcut=layer_R
    layer_R=Activation('relu')(layer_R)
    merge=Concatenate(axis=-1)([layer_left,layer_R])
    layer_R=Conv2D(filters=n_filters,kernel_size=ker_size_2,strides=strides_2,padding='same',kernel_regularizer=regularizers.l2(reg), kernel_initializer='random_normal')(merge)
    layer_R=BatchNormalization(axis=-1)(layer_R)  
    layer_R=Activation('relu')(layer_R)
    layer_R=SpatialDropout2D(0.2)(layer_R)
    layer_R=Conv2D(filters=n_filters,kernel_size=ker_size_3,strides=strides_3,padding='same',kernel_regularizer=regularizers.l2(reg), kernel_initializer='random_normal')(layer_R)
    layer_R=BatchNormalization(axis=-1)(layer_R)  
    layer_R=Add()([layer_R,layer_R_shortcut])
    layer_R=Activation('relu')(layer_R)
    layer_R=SpatialDropout2D(0.2)(layer_R)

    return layer_R


# In[42]:


def vnet(input_size):
    '''This function builds the network without compiling it.
    Parameters
    ---------------------
    input_size : tuple , size of the input
    reg : float , regularization parameters (L2)
    '''
    inputs=Input(shape=(input_size)) ## (650,650,1)
    Level_1_L = unet_layer_left(inputs, n_filters=16, ker_size_1=1,strides_1=1,ker_size_2=3, strides_2=1, ker_size_3=1, strides_3=1, reg = 0.1)
    Level_2_L = unet_layer_left(Level_1_L, n_filters=32,ker_size_1=2, strides_1=2, ker_size_2=3,strides_2=1, ker_size_3=1, strides_3=1, reg = 0.1)
    Level_3_L = unet_layer_left(Level_2_L, n_filters=64, ker_size_1=2, strides_1=2, ker_size_2=3, strides_2=1, ker_size_3=1, strides_3=1, reg = 0.1)
    Level_4_L = unet_layer_left(Level_3_L, n_filters=128, ker_size_1=2, strides_1=2, ker_size_2=3, strides_2=1, ker_size_3=1, strides_3=1, reg = 0.1)
    Level_5_L = unet_layer_left(Level_4_L, n_filters=256, ker_size_1=2, strides_1=2, ker_size_2=3, strides_2=1, ker_size_3=1, strides_3=1, reg = 0.1)
    Level_6_L = unet_layer_bottleneck(Level_5_L, n_filters= 512, ker_size_1=1, strides_1=1, ker_size_2=3, strides_2=1, reg=0.1)
    Level_5_R = unet_layer_right(Level_6_L, Level_5_L, n_filters=256, ker_size_1=1, strides_1=1, output_pad=None, ker_size_2=3, strides_2=1, ker_size_3=3, strides_3=1, reg=0.1)
    Level_4_R = unet_layer_right(Level_5_R, Level_4_L, n_filters=128, ker_size_1=2, strides_1=2, output_pad=None,ker_size_2=3, strides_2=1, ker_size_3=1, strides_3=1, reg = 0.1)
    Level_3_R = unet_layer_right(Level_4_R, Level_3_L, n_filters=64, ker_size_1=2, strides_1=2, output_pad=None, ker_size_2=3, strides_2=1, ker_size_3=1, strides_3=1, reg = 0.1)
    Level_2_R = unet_layer_right(Level_3_R, Level_2_L, n_filters=32, ker_size_1=2, strides_1=2, output_pad=None, ker_size_2=3, strides_2=1, ker_size_3=1, strides_3=1, reg = 0.1)
    Level_1_R = unet_layer_right(Level_2_R, Level_1_L, n_filters=16, ker_size_1=2, strides_1=2, output_pad=None, ker_size_2=3, strides_2=1, ker_size_3=1, strides_3=1, reg = 0.1)
    output=Conv2D(filters=1,kernel_size=1,strides=1, activation = 'sigmoid', kernel_regularizer=regularizers.l2(0.1))(Level_1_R)
    model=Model(inputs=inputs,outputs=output)
    return model


# In[44]:


class InstanceNormalization(Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# In[27]:


def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))  ## l1 regularization. Implement per-layer as kernel_regularizer=l1_reg


    

def crossentropy_loss(y_true,y_pred):
    weight_matrix=K.ones(K.shape(K.flatten(y_pred[:,:,:,:,-1]))) ## Useful when implementing weighted loss
    E=-K.expand_dims(K.log(K.flatten(tf.math.reduce_sum(tf.multiply(y_true,y_pred),-1))),axis=-1)
    return E[:,0]


def evaluation_metric_J(y_true, y_pred): ## Probabilistic Jaccard
    acc=0
    for j in range(2):
        elements_per_class=tf.math.reduce_sum(y_true[:,:,:,:,j])
        predicted_per_class=tf.math.reduce_sum(y_pred[:,:,:,:,j])
        intersection=tf.math.reduce_sum(tf.math.multiply(y_pred[:,:,:,:,j],y_true[:,:,:,:,j]))
        union=elements_per_class+predicted_per_class-intersection
        acc+=intersection/(union+0.000001)
    return acc/2

def dice_loss(y_true, y_pred): ## Probabilistic Dice
    acc=0
    for j in range(2):
        elements_per_class=tf.math.reduce_sum(y_true[:,:,:,:,j])
        predicted_per_class=tf.math.reduce_sum(y_pred[:,:,:,:,j])
        intersection=tf.math.scalar_mul(2.0,tf.math.reduce_sum(tf.math.multiply(y_pred[:,:,:,:,j],y_true[:,:,:,:,j])))
        union=elements_per_class+predicted_per_class
        acc+=(intersection+0.001)/(union+0.001)
    return 1.0-acc/2

def evaluation_metric(y_true, y_pred): ## Probabilistic Dice
    acc=0
#    for j in range(2):
    elements_per_class=tf.math.reduce_sum(y_true[:,:,:,:,1])
    predicted_per_class=tf.math.reduce_sum(y_pred[:,:,:,:,1])
    intersection=tf.math.scalar_mul(2.0,tf.math.reduce_sum(tf.math.multiply(y_pred[:,:,:,:,1],y_true[:,:,:,:,1])))
    union=elements_per_class+predicted_per_class
    acc=(intersection+0.0001)/(union+0.0001)
    return acc

def dice_loss_fore(y_true, y_pred): ## Probabilistic Dice
    elements_per_class=tf.math.reduce_sum(y_true[:,:,:,:,1])
    predicted_per_class=tf.math.reduce_sum(y_pred[:,:,:,:,1])
    intersection=tf.math.scalar_mul(2.0,tf.math.reduce_sum(tf.math.multiply(y_pred[:,:,:,:,1],y_true[:,:,:,:,1])))
    union=elements_per_class+predicted_per_class
    acc=(intersection+0.0001)/(union+0.0001)
    return 1.0-acc

def dice_ce_fore(y_true, y_pred): ## Probabilistic Dice
    dice_loss = dice_loss_fore(y_true, y_pred)
    ce_loss = crossentropy_loss(y_true, y_pred)
    return dice_loss+ce_loss

def dice_coefficient(y_true, y_pred, squared=False, smooth=1e-8):
    y_true_flat, y_pred_flat = K.flatten(y_true), K.flatten(y_pred)
    dice_nom = 2 * K.sum(y_true_flat * y_pred_flat)
    if squared:
        dice_denom = K.sum(K.square(y_true_flat) + K.square(y_pred_flat)) # squared form
    else:
        dice_denom = K.sum(K.abs(y_true_flat) + K.abs(y_pred_flat)) # abs form
    dice_coef = (dice_nom + smooth) / (dice_denom + smooth)
    return dice_coef
    
def dice_loss_internet(y_true, y_pred, squared=False, smooth=1e-8):
    dice_coef = dice_coefficient(y_true, y_pred, squared, smooth)
    return 1 - dice_coef
