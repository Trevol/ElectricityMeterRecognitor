from itertools import repeat

from counter_screen.model.CounterScreenModel import CounterScreenModel
from digits_on_screen.DigitsOnScreenModel import DigitsOnScreenModel
from utils import toInt
from utils.bbox_utils import imageByBox
from utils.detection_visualization import drawObjects
from utils.imutils import fit_image_to_shape, imshowWait, fitToWidth
from utils.iter_utils import unzip
from yolo.utils.box import visualize_boxes
import cv2
from glob import glob
import os


def evaluateOnDatasetImages():
    from utils import augmentations
    from digits_on_screen.dataset.generator import NumberImageGenerator

    augments = augmentations.make(1)
    # augments = None

    gen = NumberImageGenerator('./dataset/28x28', batchSize=8,
                               netSize=DigitsOnScreenModel.net_size, anchors=DigitsOnScreenModel.anchors,
                               augmentations=augments)
    digetsDetector = DigitsOnScreenModel(weights='./weights/6/weights_10_4.348.h5')
    for _, origBatch, augmentedBatch in gen.batches(200, DEBUG=True):
        key = 0
        for (image, _, _), (augmImage, _, _) in zip(origBatch, augmentedBatch):
            image = image.copy()
            boxes, labels, probs = digetsDetector.detect(augmImage)
            # drawObjects(image, boxes, labels, probs, (200, 0, 0))
            drawObjects(augmImage, boxes, labels, probs)
            key = imshowWait(image=(image, labels), augmImage=augmImage)
            if key == 27:
                break
        if key == 27:
            break


def main():
    # imagesPattern = '/hdd/Datasets/counters/1_from_phone/1_all_downsized/*.jpg'
    # imagesPattern = '/hdd/Datasets/counters/2_from_phone/val/*.jpg'
    # imagesPattern = '/hdd/Datasets/counters/1_from_phone/val/*.jpg'
    # imagesPattern = '/hdd/Datasets/counters/3_from_phone/*.jpg'
    # imagesPattern = '/hdd/Datasets/counters/4_from_phone/*.jpg'
    imagesPattern = '/hdd/Datasets/counters/5_from_phone/*.jpg'

    screenDetector = CounterScreenModel('../counter_screen/model/weights/2_from_scratch/weights.h5')
    digitsDetector = DigitsOnScreenModel('./weights/weights_23_0.282.h5')
    counterScreenLabel = 1
    for image_path in sorted(glob(imagesPattern)):
        image = cv2.imread(image_path)[..., ::-1]  # to RGB
        image = fit_image_to_shape(image, (1000, 1800))

        boxes, labels, probs = screenDetector.detect(image, 0.8)
        # filter only screens
        onlyScreens = ((b, l, p) for b, l, p in zip(boxes, labels, probs) if l == counterScreenLabel)
        boxes, labels, probs = unzip(onlyScreens, [], [], [])
        if len(boxes) == 0:
            continue

        screenImg = None
        for box in boxes:
            x1, y1, x2, y2 = toInt(*box)
            screenImg = imageByBox(image, box)
            screenImg = fitToWidth(screenImg, digitsDetector.net_size, 0)
            digits = digitsDetector.detect(screenImg, .5)
            print(digits)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 200), 1)

        if imshowWait(img=(image[..., ::-1], image_path), screenImg=screenImg[..., ::-1]) == 27:
            break


# main()
evaluateOnDatasetImages()
