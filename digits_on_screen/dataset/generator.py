import numpy as np
import cv2
import os
from glob import glob
from random import sample

from utils import imshowWait
from albumentations import PadIfNeeded

from utils.iter_utils import batchItems


class NumberImageGenerator:
    k = 6
    pad = 3

    def __init__(self, datasetDir, batchSize):
        self.batchSize = batchSize
        self.numberImages = self._loadNumberImages(datasetDir)

    def combineNumber(self, digits):
        return np.hstack(digits)

    @staticmethod
    def _loadNumberImages(datasetDir):
        images = []
        for n in range(9):
            numFiles = os.path.join(datasetDir, f'number_{n}', '*.png')
            numImages = [(n, cv2.imread(f)[..., ::-1]) for f in glob(numFiles)]
            images.extend(numImages)
        return numImages

    def dataItems(self):
        yield 123

    def batches(self, nBatches=None):
        if not nBatches:
            nBatches = 100
        batchItems(self.dataItems(), self.batchSize, nBatches)


if __name__ == '__main__':
    def NumberGenerator_test():
        gen = NumberImageGenerator('./28x28')
        gen


    NumberGenerator_test()
