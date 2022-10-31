import numpy as np
import tensorflow as tf
import os
from glob import glob
import tensorflow.keras.backend as K
from skimage.transform import resize
from unet_architecture import unet_arch
from skimage import exposure
import matplotlib.pyplot as plt

from PIL import Image
from conncomp import conncomp_removal

#im = Image.fromarray(A)
#im.save("your_file.jpeg")

def crop_image(img, tol=0): # funzione per croppare i bordi neri
    # img is 2D image data
    # tol  is tolerance
    try:
        mask = img>tol
        mask[img >= (1-tol)] = False

        x = np.any(mask == True, axis=0)
        y = np.any(mask == True, axis=1)

        xmin, xmax = np.where(x)[0][[0, -1]]
        ymin, ymax = np.where(y)[0][[0, -1]]

        return img[ymin:ymax,xmin:xmax]
    except:
        return img

os.environ["CUDA_VISIBLE_DEVICES"]="0"

K.clear_session()

data_path='/gpfs/ddn/fismed/COVID-19-20/CXR_challenge/lung_segmentation/dataset/Chall_subTestSet/*'
data_list = [os.path.basename(f) for f in glob(data_path)] 
predict_path = '/gpfs/ddn/fismed/COVID-19-20/CXR_challenge/lung_segmentation/dataset/Chall_subTestSet/'

dice = []

print(data_list)
model = unet_arch.vnet((512,512,1))
model.load_weights('./results/weights-val-181-0.43-0.96.hdf5')
for patient in data_list:
    img_path = predict_path + patient
    img = plt.imread(img_path)
#    img = np.array(img)

    img_name_org = './predictions_png/' + patient[:-4] +'_org.png'
    img_org = Image.fromarray((img*255)).convert('L')

    img_org.save(img_name_org)


    p5, p95 = np.percentile(img, (5, 95))
    img = exposure.rescale_intensity(img, in_range=(p5, p95))

    img = crop_image(img)

    img = resize(img,(512,512),anti_aliasing=True,cval=0,order=3)

    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    print(img.shape)
    prediction = model.predict(img)
    prediction = np.round(prediction)
    prediction = conncomp_removal.cc2d(prediction[0,:,:,0])
    img_label = Image.fromarray((prediction*255)).convert('L')
    img = Image.fromarray((img[0,:,:,0]*255)).convert('L')

    img.paste(img_label, (0,0), mask = img_label)
    img_name = './predictions_png/' + patient[:-4] +'.png'
    img.save(img_name)
