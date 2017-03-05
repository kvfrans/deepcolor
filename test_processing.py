import cv2
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from random import randint

data = glob("imgs/*.jpg")
for imname in data:

    cimg = cv2.imread(imname,1)
    cimg = np.fliplr(cimg.reshape(-1,3)).reshape(cimg.shape)
    cimg = cv2.resize(cimg, (256,256))

    img = cv2.imread(imname,0)

    # kernel = np.ones((5,5),np.float32)/25
    seg = np.ones_like(cimg)

    num_segs = 8
    seg_len = 256/num_segs

    for x in xrange(num_segs):
        for y in xrange(num_segs):
            seg[x*seg_len:(x+1)*seg_len, y*seg_len:(y+1)*seg_len, 0] = np.average(cimg[x*seg_len:(x+1)*seg_len, y*seg_len:(y+1)*seg_len, 0])
            seg[x*seg_len:(x+1)*seg_len, y*seg_len:(y+1)*seg_len, 1] = np.average(cimg[x*seg_len:(x+1)*seg_len, y*seg_len:(y+1)*seg_len, 1])
            seg[x*seg_len:(x+1)*seg_len, y*seg_len:(y+1)*seg_len, 2] = np.average(cimg[x*seg_len:(x+1)*seg_len, y*seg_len:(y+1)*seg_len, 2])


    # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_edge = cv2.adaptiveThreshold(img, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     blockSize=9,
                                     C=2)
    # img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    # img_cartoon = cv2.bitwise_and(img, img_edge)

    plt.subplot(131),plt.imshow(cimg)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(132),plt.imshow(seg)
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(133),plt.imshow(img_edge,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()
