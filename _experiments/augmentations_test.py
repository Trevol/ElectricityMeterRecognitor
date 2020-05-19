from albumentations import RGBShift
import cv2
import numpy as np

from utils.imutils import zeros, imshowWait


def main():
    image = zeros([200, 200, 3])
    image[75:125, 75:125] = 255
    transform = RGBShift(40, 40, 40, always_apply=True)
    while True:
        transformed = transform(image=image)['image']
        if imshowWait(image, transformed) == 27: break


main()
