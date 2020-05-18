from functools import partial
from itertools import repeat
from typing import Tuple, List, Iterable, Generator

import numpy as np
import cv2
import os
from glob import glob
from random import sample, choices

from albumentations import BboxParams, Compose

from digits_on_screen import DigitsOnScreenModel
from digits_on_screen.DigitsOnScreenModel import DigitsOnScreenModel
from digits_on_screen.dataset.generator import NumberImageGenerator
from utils.iter_utils import batchItems, unzip
from utils.imutils import hStack


def NumberGenerator_test():
    from utils.imutils import imshowWait
    from utils import augmentations

    gen = NumberImageGenerator('./28x28', batchSize=8,
                               netSize=DigitsOnScreenModel.net_size, anchors=DigitsOnScreenModel.anchors,
                               augmentations=augmentations.make(.7))

    for (yoloImagesBatch, y1Batch, y2Batch, y3Batch), origBatch, augmentedBatch in gen.batches(200, DEBUG=True):
        key = 0
        for (image, boxes, labels), (augmImage, augmBoxes, augmLabels) in zip(origBatch, augmentedBatch):
            image = image.copy()
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(image, (x1, y1), (x2, y2), (200, 0, 0), 1)
            for x1, y1, x2, y2 in augmBoxes:
                cv2.rectangle(augmImage, (int(x1), int(y1)), (int(x2), int(y2)), (200, 0, 0), 1)
            key = imshowWait(image=(image, labels), augmImage=augmImage)
            if key == 27:
                break
        if key == 27:
            break

    for yoloImagesBatch, y1Batch, y2Batch, y3Batch in gen.batches(2):
        print("------------------------")
        print(yoloImagesBatch.max())
        print(y1Batch.shape, y2Batch.shape, y3Batch.shape)


NumberGenerator_test()
