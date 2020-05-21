import utils.suppressTfWarnings
from digits_on_screen.DigitsOnScreenModel import DigitsOnScreenModel
from utils import toInt
from utils.detection_visualization import drawSeparateObjects, drawObjects
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
    imagesPattern = './test_images/screen1*.png'

    digitsDetector = DigitsOnScreenModel.createWithLastWeights()
    for image_path in sorted(glob(imagesPattern)):
        image = cv2.imread(image_path)

        pad = 40
        h, w = imSize(image)
        biggerImage = fill([h + pad * 2, w + pad * 2, 3], 0)
        biggerImage[pad:h + pad, pad:w + pad] = image
        image = biggerImage

        boxes, labels, probs = digitsDetector.detect(image, .5)

        displayImage = drawSeparateObjects(image, boxes, labels, probs)

        if imshowWait(img=(image, image_path), displayImage=displayImage) == 27:
            break


def detect_on_single_dataset_image():
    imageFile = '/home/trevol/Repos/Digits_Detection/not_notMNIST/Demo/Numeric/28x28/numeric_8/ArialNarrowI.png'
    digitsDetector = DigitsOnScreenModel.createWithLastWeights()
    # 1) 28x28 black on white
    # 2) padded 28x28 black on white
    # 3) 28x28 white on black
    # 2) padded 28x28 white on black
    image = cv2.imread(imageFile)
    numOfImages = 6
    image = np.hstack([image] * numOfImages)

    image = 255 - image

    pad = 5
    fillValue = image[0, 0, 0]
    bigger = fill([imHeight(image) + pad * 2, imWidth(image) + pad * 2, 3], fillValue)
    bigger[pad:imHeight(image) + pad, pad:imWidth(image) + pad] = image
    image = bigger

    boxes, labels, probs = digitsDetector.detect(image, .5)
    drawObjects(image, boxes, labels, probs)
    imshowWait(img=image)








# detect_on_single_dataset_image()
main()
