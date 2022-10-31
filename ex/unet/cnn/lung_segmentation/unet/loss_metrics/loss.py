from keras import backend as K
import tensorflow as tf

def dice_loss(y_true, y_pred): ## Probabilistic Dice computed on both background and foreground
    acc=0
    elements_per_class=tf.math.reduce_sum(y_true[:,:,:,:])
    predicted_per_class=tf.math.reduce_sum(y_pred[:,:,:,:])
    intersection=tf.math.scalar_mul(2.0,tf.math.reduce_sum(tf.math.multiply(y_pred[:,:,:,:],y_true[:,:,:,:])))
    union=elements_per_class+predicted_per_class
    acc+=intersection/(union+0.000001)
    return 1.0-acc

def dice_loss_fore(y_true, y_pred): ## Probabilistic Dice
    elements_per_class=tf.math.reduce_sum(y_true[:,:,:,:,1])
    predicted_per_class=tf.math.reduce_sum(y_pred[:,:,:,:,1])
    intersection=tf.math.scalar_mul(2.0,tf.math.reduce_sum(tf.math.multiply(y_pred[:,:,:,:,1],y_true[:,:,:,:,1])))
    union=elements_per_class+predicted_per_class
    acc=(intersection+0.0001)/(union+0.0001)
    return 1.0-acc

def surface_loss(true,pred):
    b_true = 1 - true
    b_pred = 1- pred
    f_true = true
    f_pred = pred

    true_map = b_true - f_true
    multiplied = f_pred * true_map

    return tf.math.reduce_mean(multiplied)

