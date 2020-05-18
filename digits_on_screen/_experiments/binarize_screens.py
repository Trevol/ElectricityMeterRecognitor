import cv2
import numpy as np
from glob import glob

from skimage.filters import threshold_sauvola

from utils.imutils import imshowWait


def binarize(gray, windowSize=41, k=.1):
    gray = cv2.medianBlur(gray, 3)
    gray = 255 - gray
    thresh_sauvola = threshold_sauvola(gray, window_size=windowSize, k=k)

    binary_sauvola = gray > thresh_sauvola
    binary_sauvola = np.uint8(binary_sauvola * 255)
    return binary_sauvola


def main():
    images = '/test_images/screens/*/*.png'
    f = sorted(glob(images))[0]
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    denoised = cv2.medianBlur(img, 3)

    windowSize = 41
    while True:
        binarized = binarize(img, windowSize=windowSize, k=.1)
        key = imshowWait(img, denoised, (binarized, windowSize))
        if key == 27:
            break
        elif key == ord('w'):
            windowSize += 2
        elif key == ord('s'):
            windowSize -= 2


main()
