import scipy.misc
import numpy as np
import random
import tensorflow as tf
import cPickle
import cv2
import os

def get_image(image_path):
    return transform(imread(image_path))

def transform(image, npx=512, is_crop=True):
    # npx : # of pixels width/height of image
    # if is_crop:
    #     cropped_image = center_crop(image, npx)
    # else:
    cropped_image = cv2.resize(image, (256,256))

    return np.array(cropped_image)

def imread(path):
    readimage = cv2.imread(path, 1)
    return readimage

def merge_color(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 1))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img[:,:,0]

def unpickle(file):
  fo = open(file, 'rb')
  dict = cPickle.load(fo)
  fo.close()
  return dict

def ims(name, img):
    # print img[:10][:10]
    # scipy.misc.toimage(img, cmin=0, cmax=1).save(name)
    cv2.imwrite(name, img*255)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
