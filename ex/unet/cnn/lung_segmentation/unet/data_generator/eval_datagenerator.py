from tensorflow.keras.utils import Sequence
import numpy as np
import os
import tensorflow as tf

class DataGenerator(Sequence):

    ''' Datagenerator for dd_vnet:
    Parameters---------------------
    path : path to input images
    list_X : list, indexes to call training data
    batch_size : integer, number of images per batch
    dim : tuple, integer, size of input images
    shuffle: boolean, shuffle=True means data are shuffled during calling.'''

    def __init__(self, path, list_X=list(range(1,193)), batch_size=1, dim=(650,650), shuffle=True):
        'Initialization'
        self.dim=dim
        self.batch_size = batch_size
        self.list_X = list_X
        self.path = path
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_X_temp = [self.list_X[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_X_temp, self.path)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_X_temp, path):
        'Generates data containing batch_size samples' 
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 1))
        y = np.empty((self.batch_size, *self.dim, 1))

        # Generate data
        for i, j in enumerate(list_X_temp):
            # Store sample
            arr=np.load(os.path.join(path, str(j)))
            img = np.zeros((650,650))
            seg = np.zeros((650,650))
            img[:,:]=arr[:,:,0]
            seg[:,:]=arr[:,:,1]

            X[i] = np.expand_dims(img, axis=-1)
            y[i] = np.expand_dims(seg, axis=-1)

            #X[i,] =arr # np.stack((arr,arr,arr,arr),axis=-1) ## Use here the function which creates the 4 contrasts

            # Store class
            #y[i,] = seg

        return X, y
