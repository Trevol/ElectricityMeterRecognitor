from itertools import repeat
from typing import Tuple, List, Iterable, Generator

import numpy as np
import cv2
import os
from glob import glob
from random import sample, choices

from utils.iter_utils import batchItems, unzip
from utils.imutils import hStack


class NumberImageGenerator:
    k = 6
    hPad, wPad, middlePad = 48, 48, 10

    def __init__(self, datasetDir, batchSize, augmentations, DEBUG_MODE=False):
        self.DEBUG_MODE = DEBUG_MODE
        self.batchSize = batchSize
        self.augmentations = augmentations
        self.numberImages = self.utils.load(datasetDir)
        self._imageH, self._imageW = self.numberImages[0][1].shape[:2]  # items[0].image[height, width]

    class utils:
        @staticmethod
        def imread(path, invert):
            img = cv2.imread(path)
            if invert:
                img = np.subtract(255, img, out=img)
            return img

        @classmethod
        def load(cls, datasetDir):
            images = []
            for n in range(10):
                numFiles = os.path.join(datasetDir, f'numeric_{n}', '*.png')
                # todo: pad digit image after reading??
                numImages = [(n, cls.imread(f, True)) for f in glob(numFiles)]
                images.extend(numImages)
            return images

        @staticmethod
        def toAnnotatedNumberImage(labeledDigits: List[Tuple[int, np.ndarray]],
                                   hPad, vPad, middlePad):
            """
            Combines labeled digits (list of (label, image)) to number image with digits annotations
            """
            labels, digits = unzip(labeledDigits)
            numberImage, boxes = hStack(digits, (hPad, vPad, middlePad), fillValue=0)
            boxes = [(x1, y1 - 3, x2, y2 + 3) for x1, y1, x2, y2 in boxes]
            return numberImage, boxes, labels

    def batches(self, nBatches=None):
        digitsSampler = (choices(self.numberImages, k=self.k) for _ in repeat(None))
        annotatedNumberGen = (self.utils.toAnnotatedNumberImage(labeledDigits, self.hPad, self.wPad, self.middlePad)
                              for labeledDigits in
                              digitsSampler)
        annotatedNumberBatches = batchItems(annotatedNumberGen, self.batchSize, nBatches or 100)
        # unzip to separate - imagesBatch, boxesBatch, labelsBatch
        imagesBatch_boxesBatch_labelsBatch_gen = (unzip(b) for b in annotatedNumberBatches)
        return imagesBatch_boxesBatch_labelsBatch_gen

    def __call__(self, nBatches=None):
        return self.batches(nBatches)


if __name__ == '__main__':
    from utils.imutils import imshowWait
    from utils import augmentations


    def NumberGenerator_test():
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
