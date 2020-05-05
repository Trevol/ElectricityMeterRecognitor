from itertools import cycle

import cv2
from PIL import Image
import numpy as np
import glob
from utils import imshow, fit_image_to_shape


def cvImread(filename):
    return cv2.imread(filename)


def pilImread(filename):
    return np.asarray(Image.open(filename))


def main():
    imreadMode = 'opencv'
    readFns = dict(opencv=cvImread, pil=pilImread)
    imgFiles = sorted(glob.glob('/home/trevol/hdd/Datasets/counters/2_from_phone/00*.jpg'))
    for imgFile in cycle(imgFiles):
        img = readFns[imreadMode](imgFile)
        print(img.shape)
        imshow(img=(fit_image_to_shape(img, (1024, 1024)), [imreadMode, imgFile]))
        if cv2.waitKey() == 27:
            break


main()
