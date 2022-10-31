import numpy as np
import tensorflow as tf
import os
from glob import glob
import tensorflow.keras.backend as K

from unet_architecture import unet_arch
from loss_metrics.metrics import evaluation_metric
import surface_distance.metrics as sd
from PIL import Image


#im = Image.fromarray(A)
#im.save("your_file.jpeg")



os.environ["CUDA_VISIBLE_DEVICES"]="0"

K.clear_session()

def surface_dice(true,pred,mm_tolerance = 10, spacing_mm=[1.,1.]):
    true = np.round(true)
    pred = np.round(pred) 

    true = true.astype(np.bool)
    pred = pred.astype(np.bool)

    # calcolo della distanza
    surface_distances = sd.compute_surface_distances(
        true, pred, spacing_mm )
    return sd.compute_surface_dice_at_tolerance(surface_distances,mm_tolerance)


data_path='/gpfs/ddn/fismed/COVID-19-20/CXR_challenge/lung_segmentation/dataset/Shenzen_512512/*'
data_list = [os.path.basename(f) for f in glob(data_path)] 
predict_path = '/gpfs/ddn/fismed/COVID-19-20/CXR_challenge/lung_segmentation/dataset/Shenzen_512512/'
test_list = data_list[616:]

dice = []
sdsc = []
print(test_list)
model = unet_arch.vnet((512,512,1))
model.load_weights('./results_II/results/weights-val-181-0.43-0.96.hdf5')
for patient in test_list:
    data = np.load(predict_path + patient)
    img = np.expand_dims(data[:,:,0], axis=-1)
    img = np.expand_dims(img, axis=0)

    print(img.shape)
    label = data[:,:,1]
    label = np.expand_dims(data[:,:,1], axis=-1)
    label = np.expand_dims(label, axis=0)


    prediction = model.predict(img)
    prediction = np.round(prediction)
    print(prediction.shape)
    dice.append(evaluation_metric(label.astype('float32'), prediction.astype('float32')))
    sdsc.append(surface_dice(label[0,:,:,0],prediction[0,:,:,0]))
    img_label = Image.fromarray((prediction[0,:,:,0]*255)).convert('L')
    img = Image.fromarray((img[0,:,:,0]*255)).convert('L')

    img.paste(img_label, (0,0), mask = img_label)
    img_name = './prediction_png/' + patient[:-4] +'.png'
    img.save(img_name)


avg_dice = np.mean(dice)
std_dice = np.std(dice)
avg_sdsc = np.mean(sdsc)
std_sdsc = np.std(sdsc)
print(avg_dice, std_dice, avg_sdsc, std_sdsc) 
