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
    gen = NumberImageGenerator('./28x28', batchSize=8, augmentations=augmentations.make(.7))
    gen.DEBUG_MODE = True

    # scale = .055, .060  # scale = .05, .1
    # a = IAAPerspective(scale=scale, keep_size=False, always_apply=True)
    a = augmentations.make(.7)

    for imagesBatch, boxesBatch, labelsBatch in gen.batches(200):
        for image, boxes, labels in zip(imagesBatch, boxesBatch, labelsBatch):
            transformedImg = a(image=image)['image']
            image = image.copy()
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(image, (x1, y1), (x2, y2), (200, 0, 0), 1)
            if imshowWait(image=(image, labels), transformedImg=transformedImg) == 27: return

    return
    # in DEBUG_MODE generator also yields (image, bboxes, labels)
    for xs, ys1, ys2, ys3, image, boxes, labels in gen.batches(3):
        for box, label in zip(boxes, labels):
            print(box, label)


NumberGenerator_test()
