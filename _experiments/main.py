import cv2
import numpy as np

from utils import imshow
from utils.iter_utils import unzip


def makeBatch(batchId, batchSize):
    return [(batchId, batchId * 10 + batchId, batchId * 100 + batchId * 10 + batchId, b) for b in range(batchSize)]


def main():
    nBatches = 3
    batchSize = 2
    batches = [makeBatch(b, batchSize) for b in range(1, nBatches + 1)]

    for batch in batches:
        print("----------------")
        for item in batch:
            print(item)
    print("==============================")
    unzippedBatches = [unzip(batch) for batch in batches]
    for batch in unzippedBatches:
        for item in batch:
            print(item)

main()
