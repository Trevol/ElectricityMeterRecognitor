from functools import partial
from itertools import repeat
from typing import Tuple, List, Iterable, Generator

import numpy as np
import cv2
import os
from glob import glob
from random import sample, choices

from albumentations import BboxParams, Compose

from digits_on_screen.dataset.generator import NumberImageGenerator
from utils.iter_utils import batchItems, unzip
from utils.imutils import hStack


def NumberGenerator_test():
    from utils.imutils import imshowWait
    from utils import augmentations
    anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
    gen = NumberImageGenerator('./28x28', batchSize=8, netSize=320, anchors=anchors,
                               augmentations=augmentations.make(.7))

    for (yoloImagesBatch, y1Batch, y2Batch, y3Batch), origBatch, augmentedBatch in gen.batches(200, DEBUG=True):
        for (image, boxes, labels), (augmImage, augmBoxes, augmLabels) in zip(origBatch, augmentedBatch):
            image = image.copy()
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(image, (x1, y1), (x2, y2), (200, 0, 0), 1)
            for x1, y1, x2, y2 in augmBoxes:
                cv2.rectangle(augmImage, (int(x1), int(y1)), (int(x2), int(y2)), (200, 0, 0), 1)
            if imshowWait(image=(image, labels), augmImage=augmImage) == 27: return

    return
    # in DEBUG_MODE generator also yields (image, bboxes, labels)
    for xs, ys1, ys2, ys3, image, boxes, labels in gen.batches(3):
        for box, label in zip(boxes, labels):
            print(box, label)


NumberGenerator_test()
