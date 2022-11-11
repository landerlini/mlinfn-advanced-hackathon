#!/usr/bin/env python
# coding: utf-8

# # Definition of loss functions and metrics

# The problem we want to solve is a segmentation problem. For this reason we should use a correct loss function and metric to both optimize and evaluate the training. 
# 
# As regards the loss function, we are going to use the Dice Similarity Coefficient (DSC) that measures how much the true and the predicted areas overlap. DSC is defined this way:
# 
# $ \begin{align}
# DSC(M_{true}, M_{pred}) = 2 \cdot \frac{M_{true} \cdot M_{pred}}{M_{true} + M_{pred}}
# \end{align}
# $
# 
# Visually:
# 
# <img src="../images/DSC.png" alt="" class="bg-primary mb-1" width="500px">
# 
# If there is a perfect overlapping, the DSC will be equal to 1. If there is no overlapping, DSC will be equal to 0. Since we want to use the DSC as a loss function, we want that the better overlapping is nearer to 0. So we define the DSC loss funtion in this way:
# 
# $ \begin{align}
# DSC_{loss}(M_{true}, M_{loss}) = 1 - 2 \cdot \frac{M_{true} \cdot M_{pred}}{M_{true} + M_{pred}}
# \end{align}
# $

# ## **Let's write this loss function with Tensorflow**

# In[2]:


import tensorflow as tf
import numpy as np

import Loss_Metrics.surface_distance.metrics as sd

def DSC_loss(y_true, y_pred): ## Probabilistic Dice
    elements_per_class=tf.math.reduce_sum(y_true) # we sum all the pixels belonging to lungs on the labels
    predicted_per_class=tf.math.reduce_sum(y_pred) # we sum all the pixels belonging to lungs on predicted mask
    # We compute the intersection: we multiply the matrices of the predicted and the true masks.
    # Then we sum the elements of the resulting matrix to obtain the number of overlapping pixels;
    # Lastly, we multiply this result by 2.
    intersection=tf.math.scalar_mul(2.0,tf.math.reduce_sum(tf.math.multiply(y_pred,y_true)))
    union=elements_per_class+predicted_per_class
    acc=(intersection+0.0001)/(union+0.0001) # the correction is needed to not let the algorithm loss going to inf.
    return 1.0-acc


# We can use many different loss function to solve this problem but for now DSC loss is ok.
# In the same way, we can define many matrics to measure the performance of our algorithm. For now, we will see only the DSC explained above and the Jaccard coefficient. The Jaccard index or Jaccard Similarity Coefficient is an index that measures the similarity between two samples. It is simply defined as the intersection over union.

# In[3]:


def Jaccard(y_true, y_pred): ## Probabilistic Jaccard computed on both background and foreground (It has not been used yet)
    acc=0
    y_pred = tf.math.round(y_pred)
    elements_per_class=tf.math.reduce_sum(y_true)
    predicted_per_class=tf.math.reduce_sum(y_pred)
    intersection=tf.math.reduce_sum(tf.math.multiply(y_pred,y_true))
    union=elements_per_class+predicted_per_class-intersection
    acc+=intersection/(union+0.000001)
    return acc/2

def DSC(y_true, y_pred): ## Dice coefficient computed only on the foreground to evaluate performances on GG segmentation.
    acc=0
    y_pred = tf.math.round(y_pred) # NB: we want to obtain a binary mask but we cannot perform the
    # rounding operation in the loss function. Why?
    elements_per_class=tf.math.reduce_sum(y_true) # we sum all the pixels belonging to lungs on the labels
    predicted_per_class=tf.math.reduce_sum(y_pred) # we sum all the pixels belonging to lungs on predicted mask
    # We compute the intersection: we multiply the matrices of the predicted and the true masks.
    # Then we sum the elements of the resulting matrix to obtain the number of overlapping pixels;
    # Lastly, we multiply this result by 2.
    intersection=tf.math.scalar_mul(2.0,tf.math.reduce_sum(tf.math.multiply(y_pred,y_true)))
    union=elements_per_class+predicted_per_class
    acc=(intersection+0.0001)/(union+0.0001) # the correction is needed to not let the algorithm loss going to inf.
    return acc

def border_dice(true,pred,mm_tolerance = 2, spacing_mm=[1.,1.]):
    true = np.round(true)
    pred = np.round(pred) 

    true = true.astype(np.bool)
    pred = pred.astype(np.bool)

    # calcolo della distanza
    surface_distances = sd.compute_surface_distances(
        true, pred, spacing_mm )
    return sd.compute_surface_dice_at_tolerance(surface_distances,mm_tolerance)
# test per capire metriche e loss
