import cv2
import numpy as np
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

from utils.imutils import imshowWait


def main():
    image = cv2.imread("./counter_images/01305.png", cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread("../venv/lib/python3.7/site-packages/skimage/data/page.png", cv2.IMREAD_GRAYSCALE)

    image = 255 - image

    image = cv2.medianBlur(image, 7)

    binary_global = image > threshold_otsu(image)
    binary_global = np.uint8(binary_global * 255)

    window_size = 41
    thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.8)
    thresh_sauvola = threshold_sauvola(image, window_size=window_size, k=.1)

    binary_niblack = image > thresh_niblack
    binary_niblack = np.uint8(binary_niblack * 255)

    binary_sauvola = image > thresh_sauvola
    binary_sauvola = np.uint8(binary_sauvola * 255)

    imshowWait(image=image, binary_global=binary_global, binary_niblack=binary_niblack, binary_sauvola=binary_sauvola)


main()
