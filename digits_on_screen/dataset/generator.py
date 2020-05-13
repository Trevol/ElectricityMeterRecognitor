from itertools import repeat
from typing import Tuple, List, Iterable, Generator

import numpy as np
import cv2
import os
from glob import glob
from random import sample, choices

from albumentations import BboxParams, Compose

from utils.iter_utils import batchItems, unzip
from utils.imutils import hStack


class NumberImageGenerator:
    k = 6
    hPad, vPad, middlePad = 48, 48, 10
    padding = hPad, vPad, middlePad

    def __init__(self, datasetDir, batchSize, augmentations, DEBUG_MODE=False):
        self.DEBUG_MODE = DEBUG_MODE
        self.batchSize = batchSize
        self.augmentations = self.utils.composeAugmentations(augmentations)
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
        def toAnnotatedNumberImage(labeledDigits, padding):
            """
            Combines labeled digits (list of (label, image)) to number image with digits annotations
            """
            labels, digits = unzip(labeledDigits)
            numberImage, boxes = hStack(digits, padding, fillValue=0)
            boxes = [(x1, y1 - 3, x2, y2 + 3) for x1, y1, x2, y2 in boxes]
            return numberImage, boxes, labels

        @staticmethod
        def composeAugmentations(augmentations):
            bbox_params = BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=.8)
            return Compose(augmentations or [], bbox_params=bbox_params)

        @staticmethod
        def augment(augmentations, image_boxes_labels):
            image, boxes, labels = image_boxes_labels
            r = augmentations(image=image, bboxes=boxes, labels=labels)
            return r['image'], r['bboxes'], r['labels']

    def batches(self, nBatches=None):
        digitsSampler = (choices(self.numberImages, k=self.k) for _ in repeat(None))

        numberImageFn = self.utils.toAnnotatedNumberImage
        annotatedNumberImagesGen = (numberImageFn(labeledDigits, self.padding) for labeledDigits in digitsSampler)

        augmentFn = self.utils.augment
        augmentedImagesGen = (augmentFn(self.augmentations, annImg) for annImg in annotatedNumberImagesGen)
        raise NotImplementedError("How to yield all this stuff???? Original, augmented, yolo-formatted...")
        annotatedNumberBatches = batchItems(augmentedImagesGen, self.batchSize, nBatches or 100)
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
