import tensorflow as tf
import tensorflow.keras 
from glob import glob
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import random
import pandas as pd
from scipy import ndimage

import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import exposure

from gray_cnn import cnn
from lung_cnn import unet_arch
from utils import utils, pre_processing, visualize, conncomp_removal, augmentation

# carico il modello per la gray cnn
input_gray = (50,50,1)
gray_model = cnn.cnn_model(input_gray)
gray_model.load_weights('./gray_cnn/weights-val-010-0.26-1.00.hdf5')
# carico la rete per la segmentazione del polmone
input_lung = (512,512,1)
lung_model = unet_arch.vnet(input_lung)
lung_model.load_weights('./lung_cnn/weights-val-283-4.06-0.95.hdf5')
# carico le features cliniche
clin_feat = pd.read_csv('clinfeat.csv', index_col='ImageFile')

path = '/gpfs/ddn/fismed/CovidRX_22/dataset/covidCXR/'
#path = '/gpfs/ddn/fismed/COVID-19-20/CXR_challenge/segmented_dataset_challenge/sbagliate/'

files = glob("/gpfs/ddn/fismed/CovidRX_22/dataset/covidCXR/*.png")
#files = glob("/gpfs/ddn/fismed/COVID-19-20/CXR_challenge/segmented_dataset_challenge/sbagliate/*.png")

aug = True
# faccio una lista di pazienti le cui segmentazioni o img sono inaccettabili
wrong_patient = ['P_2_7','P_2_11','P_2_17', 'P_2_69', 'P_2_98', 'P_2_99', 'P_49', 'P_110', 'P_159', 'P_575', 'P_581', 'P_612', 'P_805', 'P_1_34', 'P_423']

file_names = [ os.path.basename(n) for n in files ]
print(file_names)
def predict_gray(img,gray_net):
    
    # leggo i dati e li standardizzo media 0 e divido per std
    p5, p95 = np.percentile(img, (5, 95))
    img = exposure.rescale_intensity(img, in_range=(p5, p95))

    standard_img = (img - np.mean(img, axis=(0,1)))/np.std(img)
    standard_img = pre_processing.crop_image(standard_img)
    resized_img_s = resize(standard_img, (50,50), anti_aliasing=True, cval=0, order=3)
    resized_img = np.expand_dims(resized_img_s, axis =-1)
    resized_img = np.expand_dims(resized_img, axis =0)
    # faccio la predizione
    prediction = gray_net.predict(resized_img)
    tf.keras.backend.clear_session()
    return prediction

def predict_lung(img, lung_net):
    resized_img = np.expand_dims(img, axis =-1)
    resized_img = np.expand_dims(resized_img, axis =0)
    # faccio la predizione
    prediction = lung_net.predict(resized_img)
    tf.keras.backend.clear_session()
    return prediction

for patient in file_names:
    print(patient[0:-4])
    if patient[0:-4] in wrong_patient:
        print('The patient is not included in the dataset')
        continue
    else:
        current_patient = path + patient
        current_img = plt.imread(current_patient)
        pred_gray = predict_gray(current_img, gray_model)
        class_gray = np.round(pred_gray)
        if class_gray == 0:
            p5, p95 = np.percentile(current_img, (5, 95))
            img = exposure.rescale_intensity(current_img, in_range=(p5, p95))
            img = pre_processing.crop_image(img)
            img = 1 - img
        else:
            p5, p95 = np.percentile(current_img, (5, 95))
            img = exposure.rescale_intensity(current_img, in_range=(p5, p95))
            img = pre_processing.crop_image(img)

        res_img = resize(img, (512,512), anti_aliasing=True, cval=0, order=3)
        pred_lung = predict_lung(res_img, lung_model)
        pred_lung = conncomp_removal.cc2d(pred_lung[0,:,:,0])
        pred_lung=ndimage.binary_fill_holes(pred_lung).astype(int)
        img_res = resize(img,(1024,1024),anti_aliasing=True,cval=0,order=3)
        lung_res = resize(pred_lung,(1024,1024),anti_aliasing=True, preserve_range = True, cval=0,order=0)
        features = clin_feat.loc[[patient]]
        label = features.Prognosis
        features= features.drop('Prognosis', axis =1)
        features_array = np.array(features)
        label_array = np.array(label)
        print(label)
        print(label_array)
        if label_array == 1:
            utils.save_data_with_mask_and_folder(img_res, lung_res, features_array, label_array, patient, 'dataset_severe/',suffix="") 
            visualize.plot_lung(current_img, img_res, lung_res,  patient)
            if aug == True:
                aug_img, aug_mask = augmentation.flip(img_res, lung_res)
                M = 2 # number of transformations
                augment = random.sample(["zoom", "rotation", "shift"], M)
                if "zoom" in augment:
                    aug_img, aug_mask = augmentation.zoom(aug_img, aug_mask)
                if "rotation" in augment:
                    aug_img, aug_mask = augmentation.scipy_rotate(aug_img, aug_mask)
                if "shift" in augment:
                    aug_img, aug_mask = augmentation.scipy_shift(aug_img, aug_mask)
         
                aug_img = resize(aug_img,(1024,1024),anti_aliasing=True,cval=0,order=3)
                aug_mask = resize(aug_mask,(1024,1024),anti_aliasing=True, preserve_range = True, cval=0,order=0)
                print(lung_res.max())
                utils.save_data_with_mask_and_folder(aug_img, aug_mask, features_array, label_array, patient,'dataset_severe/', suffix="aug")
                visualize.plot_lung(current_img, aug_img, aug_mask,  patient, suffix='aug')

        elif label_array == 0:
            utils.save_data_with_mask_and_folder(img_res, lung_res, features_array, label_array, patient, 'dataset_mild/',suffix="")
            visualize.plot_lung(current_img, img_res, lung_res,  patient)
            if aug == True:
                aug_img, aug_mask = augmentation.flip(img_res, lung_res)
                M = 2 # number of transformations
                augment = random.sample(["zoom", "rotation", "shift"], M)
                if "zoom" in augment:
                    aug_img, aug_mask = augmentation.zoom(aug_img, aug_mask)
                if "rotation" in augment:
                    aug_img, aug_mask = augmentation.scipy_rotate(aug_img, aug_mask)
                if "shift" in augment:
                    aug_img, aug_mask = augmentation.scipy_shift(aug_img, aug_mask)

                aug_img = resize(aug_img,(1024,1024),anti_aliasing=True,cval=0,order=3)
                aug_mask = resize(aug_mask,(1024,1024),anti_aliasing=True, preserve_range = True, cval=0,order=0)
                print(lung_res.max())
                utils.save_data_with_mask_and_folder(aug_img, aug_mask, features_array, label_array, patient,'dataset_mild/', suffix="aug")
                visualize.plot_lung(current_img, aug_img, aug_mask,  patient, suffix='aug')
        else:
            print('UNDEFINED PATIENT LABEL')
