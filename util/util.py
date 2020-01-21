from __future__ import print_function

import numpy as np
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import functools


#------------------------------------------------------------------------------
# Global Settings
#------------------------------------------------------------------------------
IMG_CACHE_COUNT = 1000

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------
"""
Sub-function for loading uncropped images with caching
"""
@functools.lru_cache(maxsize=IMG_CACHE_COUNT)
def load_image_cached(path):
    extension = os.path.splitext(path)[1]

    if(extension == '.dng'):
        import rawpy
        with rawpy.imread(path) as raw:
            img = raw.postprocess()
    elif(extension=='.bmp' or extension=='.jpg' or extension=='png'):
        import cv2
        img = cv2.imread(path)[:,:,::-1]
    else:
        img = (255 * plt.imread(path)[:,:,:3]).astype('uint8')

    return img

"""
Loads an image (or a part of it) from the given path

    partOrigin:     tuple of x and y indices
    partDim:        tuple with width and height of image part
"""
def load_image(path, partOrigin=None, partDim=(0, 0)):
    img = load_image_cached(path)

    if partOrigin is not None:
        x, y = partOrigin
        w, h = partDim
        img = np.ascontiguousarray(img[x:x+w,y:y+h,:], dtype=np.uint8)

    return img

def save_image(image_numpy, image_path, ):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=255./2.):
# def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=1.):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
# def im2tensor(image, imtype=np.uint8, cent=1., factor=1.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

