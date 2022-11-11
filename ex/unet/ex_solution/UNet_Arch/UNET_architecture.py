#!/usr/bin/env python
# coding: utf-8

# # U-Net architecture definition

# We have to define the structure of our FCNN. Let's import the packages. NB: we are not importing Keras directly but we import keras through tensorflow!

# In[2]:


import numpy as np
import os
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import regularizers, initializers, constraints
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Input, BatchNormalization, SpatialDropout2D, Conv2D, Conv2DTranspose, Dropout, Concatenate, Lambda, Activation, Reshape, Add
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf


# Spieghiamo cosa fa ogni singolo layer.

# In[3]:


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


# In[4]:


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


# In[5]:


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


# In[6]:


def U_net(input_size):
    '''This function builds the network without compiling it.
    Parameters
    ---------------------
    input_size : tuple , size of the input
    reg : float , regularization parameters (L2)
    '''
    inputs=Input(shape=(input_size)) ## 
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