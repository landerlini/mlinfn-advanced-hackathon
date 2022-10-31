import numpy as np
import matplotlib.pyplot as plt

def plot_gray(or_im,pp_im,resized_im,basename):
    fig, ax = plt.subplots(1,3,tight_layout=True)

    for a in ax:
        a.axis("off")

    fig.suptitle("{}".format(basename))

    xshape = or_im.shape[0]
    yshape = or_im.shape[1]

    xy_corr = yshape / xshape

    ax[0].set_title("Original Image")
    ax[0].imshow(or_im,cmap='bone',aspect=xy_corr)
    ax[1].set_title("Pre-processed Image")
    ax[1].imshow(pp_im,cmap='bone',aspect=xy_corr)
    ax[2].set_title("Resized Pre-Processed Image")
    ax[2].imshow(resized_im,cmap='bone')

    plt.savefig("plot_corr/{}".format(basename))
    plt.close()

def plot_lung(or_im,resized_img, lung_mask, basename, suffix = ''):
    fig, ax = plt.subplots(1,3,tight_layout=True)

    for a in ax:
        a.axis("off")

    fig.suptitle("{}".format(basename))

    xshape = or_im.shape[0]
    yshape = or_im.shape[1]

    xy_corr = yshape / xshape

    ax[0].set_title("Original Image")
    ax[0].imshow(or_im,cmap='bone',aspect=xy_corr)
    ax[1].set_title("Resized Image")
    ax[1].imshow(resized_img,cmap='bone')
    ax[2].set_title("Lung Overlay Image")
    ax[2].imshow(resized_img,cmap='bone')
    ax[2].imshow(lung_mask,cmap=plt.cm.viridis, alpha = 0.5)

    plt.savefig("plot_balanced_checkdata/{}".format(suffix+basename))
    plt.close()

