import numpy as np
import tensorflow as tf
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from glob import glob

from keras import initializers, regularizers, constraints, optimizers, metrics
from keras import backend as K
from keras.activations import softmax
from keras.layers import Dense, Input, Conv2D, Conv2DTranspose, Dropout, Flatten, BatchNormalization, Concatenate, Lambda, Activation, Reshape, Layer, InputSpec
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import Sequence

import data_generator.train_datagenerator as DG
import loss_metrics.loss as dd_loss
import loss_metrics.metrics as dd_metrics
import unet_architecture.unet_arch as unet

tf.keras.backend.clear_session()

path='/gpfs/ddn/fismed/COVID-19-20/CXR_challenge/lung_segmentation/dataset/Shenzen_512512/' # input images path

checkpoint_path="./results/weights-{epoch:03d}-{loss:.2f}-{val_evaluation_metric:.2f}.hdf5" # intermediate weight save path
checkpoint_path_val="./results/weights-val-{epoch:03d}-{loss:.2f}-{val_evaluation_metric:.2f}.hdf5"  # intermediate weigth save path

# CALLBACKS DEFINITIONS

check=ModelCheckpoint(filepath=checkpoint_path, monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=5) # Checkpoints parameters
check_val=ModelCheckpoint(filepath=checkpoint_path_val, monitor='val_evaluation_metric', verbose=0, save_best_only=True, save_weights_only=True, mode='max')
adamlr = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
reduce_lr = ReduceLROnPlateau(monitor='val_evaluation_metric', factor=0.5, patience=8, min_lr=0.000005, verbose= True)

data_path='/gpfs/ddn/fismed/COVID-19-20/CXR_challenge/lung_segmentation/dataset/Shenzen_512512/*'
data_list = [os.path.basename(f) for f in glob(data_path)] 

train_list = data_list[0:545]
val_list = data_list[545:616]

training_gen = DG.DataGenerator(path=path,list_X=train_list,batch_size=8)
validation_gen = DG.DataGenerator(path=path,list_X=val_list,batch_size=1)

unetshallow=unet.vnet((512,512,1))
met = tf.keras.metrics.MeanIoU(num_classes=2)
unetshallow.compile(loss=dd_loss.dice_loss,optimizer=adamlr, metrics=[dd_metrics.evaluation_metric])
unetshallow.summary()
history=unetshallow.fit(training_gen, validation_data= validation_gen, epochs=300,callbacks=[reduce_lr, check_val, check],verbose=1)

with open('history.json', 'w') as f:
    json.dump(str(history.history), f)
