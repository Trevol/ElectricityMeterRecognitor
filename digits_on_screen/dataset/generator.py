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
    hPad, wPad, middlePad = 32, 48, 2

    def __init__(self, datasetDir, batchSize, DEBUG_MODE=False):
        self.DEBUG_MODE = DEBUG_MODE
        self.batchSize = batchSize
        self.numberImages = self.utils.load(datasetDir)
        self._imageHeight = self.numberImages[0][1].shape[0]  # items[0].image.height

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
        def toAnnotatedNumber(
                labeledDigits: List[Tuple[int, np.ndarray]],
                hPad, wPad, middlePad):
            """
            Combines labeled digits (list of (label, image)) to number image with digits annotations
            """
            labels, digits = unzip(labeledDigits)
            numberImg = np.hstack(digits)

            # todo: pad up, down, left, right and between digits
            # or images will already be padded in cache

            h, w = numberImg.shape[:2]
            if len(numberImg.shape) == 3:  # color image
                paddedShape = h + 2 * hPad, w + 2 * wPad, 3
            else:  # gray scale
                paddedShape = h + 2 * hPad, w + 2 * wPad

            paddedImage = np.full(paddedShape, 0, np.uint8)
            paddedImage[hPad:hPad + h, wPad:wPad + w] = numberImg

            boxes = [(1, 1, 5, 6) for _ in labels]
            return paddedImage, boxes, labels

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
    from utils import imshowWait, augmentations


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
