from digits_on_screen.DigitsOnScreenModel import DigitsOnScreenModel
from utils import toInt
from utils.imutils import fit_image_to_shape, imshowWait, imHeight, imWidth, imSize, fill, imChannels, fitToWidth
from utils.iter_utils import unzip
from yolo.utils.box import visualize_boxes
import cv2
from glob import glob
import os
import numpy as np


def rPad():
    # rPad = fill((imHeight(image), 20, 3), image[0, -1, 0])
    # image = np.hstack([image, rPad])
    pass


def main():
    imagesPattern = './test_images/screens/full_res/*.png'

    digitsDetector = DigitsOnScreenModel('./weights/1/weights_7_1.097.h5')
    netSize = 320  # DigitsOnScreenModel.net_size
    for image_path in sorted(glob(imagesPattern)):
        image = cv2.imread(image_path)

        fittedImg = fitToWidth(image, netSize)

        # image = fit_image_to_shape(image, (1000, 1800))

        boxes, labels, probs = digitsDetector.detect(fittedImg, .5)

        drawObjects(fittedImg, boxes, labels, probs)

        if imshowWait(img=(fittedImg, image_path)) == 27:
            break


def drawObjects(image, boxes, labels, probs):
    for i, (box, label, prob) in enumerate(zip(boxes, labels, probs)):
        x1, y1, x2, y2 = toInt(*box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 200), 1)
        center = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(image, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, .5, (200, 0, 0))
        print(i, label, prob)


main()
