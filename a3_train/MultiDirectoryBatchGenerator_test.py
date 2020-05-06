import utils.suppressTfWarnings
from a3_train.train import createDataGenerator, makeAugmentations
from utils import imshow
from yolo.config import ConfigParser
import numpy as np
import cv2


def main():
    dataDirs = [
        # '/hdd/Datasets/counters/0_from_internet/train',
        '/hdd/Datasets/counters/1_from_phone/train',
        '/hdd/Datasets/counters/2_from_phone/train'
    ]
    config = ConfigParser("configs/counters.json")
    gen = createDataGenerator(dataDirs, config, shuffle=False, augmentations=makeAugmentations(), steps_per_epoch=10)
    normalizedImgs, dd1, dd2, dd3 = gen.next_batch()
    for normalizedImg, d1, d2, d3 in zip(normalizedImgs, dd1, dd2, dd3):
        img = np.uint8(normalizedImg * 255.)
        imshow(img=img[..., ::-1])
        if cv2.waitKey() == 27:
            break


main()
