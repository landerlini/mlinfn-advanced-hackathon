import numpy as np

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

