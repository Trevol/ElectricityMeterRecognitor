from itertools import repeat
from typing import Tuple, List, Iterable, Generator

import numpy as np
import cv2
import os
from glob import glob
from random import sample

from albumentations import IAAPerspective

from utils.iter_utils import batchItems, unzip


class NumberImageGenerator:
    k = 6
    hPad, wPad, middlePad = 32, 48, 10

    def __init__(self, datasetDir, batchSize, DEBUG_MODE=False):
        self.DEBUG_MODE = DEBUG_MODE
        self.batchSize = batchSize
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

        @classmethod
        def toAnnotatedNumber___(cls,
                                 labeledDigits: List[Tuple[int, np.ndarray]],
                                 hPad, wPad, middlePad):
            """
            Combines labeled digits (list of (label, image)) to number image with digits annotations
            """
            labels, digits = unzip(labeledDigits)
            imH, imW = digits[0].shape[:2]
            hPadding = np.zeros([imH, hPad, 3], np.uint8)

            hParts = [hPadding] + digits + [hPadding]
            paddedImage = np.hstack(hParts)

            wPadding = np.zeros([wPad, paddedImage.shape[1], 3], np.uint8)
            paddedImage = np.vstack([wPadding, paddedImage, wPadding])

            boxes = [(1, 1, 5, 6) for _ in labels]
            return paddedImage, boxes, labels

        @staticmethod
        def toAnnotatedNumber(labeledDigits: List[Tuple[int, np.ndarray]],
                              hPad, vPad, middlePad):
            """
            Combines labeled digits (list of (label, image)) to number image with digits annotations
            """
            labels, digits = unzip(labeledDigits)
            numberImage, boxes = hStack(digits, (hPad, vPad, middlePad), fillValue=0)
            return numberImage, boxes, labels

    def batches(self, nBatches=None):
        digitsGen = (sample(self.numberImages, self.k) for _ in repeat(None))
        annotatedNumberGen = (self.utils.toAnnotatedNumber(labeledDigits, self.hPad, self.wPad, self.middlePad)
                              for labeledDigits in
                              digitsGen)
        annotatedNumberBatches = batchItems(annotatedNumberGen, self.batchSize, nBatches or 100)
        # unzip to separate - imagesBatch, boxesBatch, labelsBatch
        imagesBatch_boxesBatch_labelsBatch_gen = (unzip(b) for b in annotatedNumberBatches)
        return imagesBatch_boxesBatch_labelsBatch_gen

    def __call__(self, nBatches=None):
        return self.batches(nBatches)


if __name__ == '__main__':
    from utils import imshowWait, augmentations, hStack


    def NumberGenerator_test():
        gen = NumberImageGenerator('./28x28', batchSize=8)
        gen.DEBUG_MODE = True

        # scale = .055, .060  # scale = .05, .1
        # a = IAAPerspective(scale=scale, keep_size=False, always_apply=True)
        a = augmentations.make()

        for imagesBatch, boxesBatch, labelsBatch in gen.batches(200):
            for image, boxes, labels in zip(imagesBatch, boxesBatch, labelsBatch):
                transformedImg = a(image=image)['image']
                if imshowWait(image=(image, labels), transformedImg=transformedImg) == 27: return

        return
        # in DEBUG_MODE generator also yields (image, bboxes, labels)
        for xs, ys1, ys2, ys3, image, boxes, labels in gen.batches(3):
            for box, label in zip(boxes, labels):
                print(box, label)


    NumberGenerator_test()
