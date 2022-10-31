import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

def evaluation_metric_J(y_true, y_pred): ## Probabilistic Jaccard computed on both background and foreground (It has not been used yet)
    acc=0
    for j in range(2):
        elements_per_class=tf.math.reduce_sum(y_true[:,:,:,:,j])
        predicted_per_class=tf.math.reduce_sum(y_pred[:,:,:,:,j])
        intersection=tf.math.reduce_sum(tf.math.multiply(y_pred[:,:,:,:,j],y_true[:,:,:,:,j]))
        union=elements_per_class+predicted_per_class-intersection
        acc+=intersection/(union+0.000001)
    return acc/2

def evaluation_metric(y_true, y_pred): ## Dice coefficient computed only on the foreground to evaluate performances on GG segmentation.
    acc=0
    y_pred = tf.math.round(y_pred)
    elements_per_class=tf.math.reduce_sum(y_true[:,:,:,:])
    predicted_per_class=tf.math.reduce_sum(y_pred[:,:,:,:])
    intersection=tf.math.scalar_mul(2.0,tf.math.reduce_sum(tf.math.multiply(y_pred[:,:,:,:],y_true[:,:,:,:])))
    union=elements_per_class+predicted_per_class
    acc=(intersection+0.0001)/(union+0.0001)
    return acc

def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice
