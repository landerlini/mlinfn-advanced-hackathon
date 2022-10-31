import numpy as np
import pickle
 
def crop_image(img, lung_mask, tol=0): # funzione per croppare i bordi neri
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    mask[img >= (1-tol)] = False

    x = np.any(mask == True, axis=0)
    y = np.any(mask == True, axis=1)

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]

    return img[ymin:ymax,xmin:xmax], lung_mask[ymin:ymax,xmin:xmax]

def save_data_with_mask(i, lung_mask, features_array, label_array, f, suffix=""):
    input_data = [i, lung_mask, features_array, label_array]

    if suffix != "":
        suffix = "_" + suffix

    save_path = 'dataset/' + f.split('.')[0] + suffix
    open_file = open( save_path, "wb")
    pickle.dump(input_data, open_file)
    open_file.close()

def save_data_with_mask_and_folder(i, lung_mask, features_array, label_array, f, folder, suffix=""):
    input_data = [i, lung_mask, features_array, label_array]

    if suffix != "":
        suffix = "_" + suffix

    save_path = folder  + f.split('.')[0] + suffix
    open_file = open( save_path, "wb")
    pickle.dump(input_data, open_file)
    open_file.close()

