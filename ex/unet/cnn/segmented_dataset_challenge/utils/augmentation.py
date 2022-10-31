import random
import numpy as np
from scipy import ndimage
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path,"../../"))

def scipy_rotate(image, mask):
    # define some rotation angles
    angles = [-30,-15, -10, 10, 15, 30]
    # pick angles at random
    angle = random.choice(angles)
    # rotate volume
    image = ndimage.rotate(image, angle, cval=0, reshape=False) #mode='nearest'
    mask = ndimage.rotate(mask, angle, cval=0, reshape=False)
    return image, mask
    
def scipy_shift(img,mask):
    # define some y-displacemets (in percentage of pixels)
    list_shift = [-0.078, -0.058, -0.038, 0.038, 0.058, 0.078]
    # pick shift at random
    ty = random.choice(list_shift)*img.shape[1]
    tx = random.choice(list_shift)*img.shape[0]
    # shift volume
    img_shift = ndimage.shift(img, shift=(tx,ty), order=3, cval=0)
    mask_shift = ndimage.shift(mask, shift=(tx,ty), order=0, cval=0)
    return img_shift, mask_shift

def zoom(img,mask):
    x,y = img.shape
    list_zoom = [0.05,0.04,0.03,0.02]
    m = random.choice(list_zoom)
    dx = int(m*x)
    dy = int(m*y)

    xm = dx
    xM = x - dx

    ym = dy
    yM = y - dy

    if xm < 0:
        xm=0
    if ym < 0:
        ym=0
    if xM > img.shape[0]:
        xM=img.shape[0]
    if yM > img.shape[1]:
        yM=img.shape[1]
    xm=int(xm)
    ym=int(ym)
    xM=int(xM)
    yM=int(yM)
    im = img[xm:xM,ym:yM]
    mask = mask[xm:xM,ym:yM]
    return im, mask

def flip(img,mask):
    return img[:,::-1], mask[:,::-1]
