import cv2
import numpy as np
from glob import glob

from skimage.filters import threshold_sauvola

from utils.imutils import imshowWait, binarizeSauvola


def main():
    images = './test_images/screen1.png'
    f = sorted(glob(images))[0]
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    denoised = cv2.medianBlur(img, 3)

    windowSize = 41
    while True:
        binarized = binarizeSauvola(255-img, windowSize=windowSize, k=.1)
        key = imshowWait(img, denoised, (binarized, windowSize), 255-binarized)
        if key == 27:
            break
        elif key == ord('w'):
            windowSize += 2
        elif key == ord('s'):
            windowSize -= 2


main()
