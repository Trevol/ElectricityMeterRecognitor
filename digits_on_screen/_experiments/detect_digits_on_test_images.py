import utils.suppressTfWarnings
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
    imagesPattern = './test_images/screen1*.png'

    digitsDetector = DigitsOnScreenModel('./weights/4_resize_finetune/weights_28_0.150.h5')
    for image_path in sorted(glob(imagesPattern)):
        image = cv2.imread(image_path)

        pad = 40
        h, w = imSize(image)
        biggerImage = fill([h + pad * 2, w + pad * 2, 3], 0)
        biggerImage[pad:h + pad, pad:w + pad] = image
        image = biggerImage

        boxes, labels, probs = digitsDetector.detect(image, .5)

        drawObjects(image, boxes, labels, probs)

        if imshowWait(img=(image, image_path)) == 27:
            break


def detect_on_single_dataset_image():
    imageFile = '/home/trevol/Repos/Digits_Detection/not_notMNIST/Demo/Numeric/28x28/numeric_8/ArialNarrowI.png'
    digitsDetector = DigitsOnScreenModel('./weights/4_resize_finetune/weights_28_0.150.h5')
    # 1) 28x28 black on white
    # 2) padded 28x28 black on white
    # 3) 28x28 white on black
    # 2) padded 28x28 white on black
    image = cv2.imread(imageFile)
    numOfImages = 6
    image = np.hstack([image] * numOfImages)

    image = 255 - image

    pad = 80
    fillValue = image[0, 0, 0]
    bigger = fill([imHeight(image) + pad * 2, imWidth(image) + pad * 2, 3], fillValue)
    bigger[pad:imHeight(image) + pad, pad:imWidth(image) + pad] = image
    image = bigger

    boxes, labels, probs = digitsDetector.detect(image, .5)
    drawObjects(image, boxes, labels, probs)
    imshowWait(img=image)


def drawObjects(image, boxes, labels, probs):
    print("-----Objects------")
    for i, (box, label, prob) in enumerate(zip(boxes, labels, probs)):
        x1, y1, x2, y2 = toInt(*box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 200), 1)
        center = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(image, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, .5, (200, 0, 0))
        print(i, label, prob)


# detect_on_single_dataset_image()
main()
