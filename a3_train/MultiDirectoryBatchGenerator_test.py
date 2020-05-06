from tqdm import tqdm

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
    # gen = createDataGenerator(dataDirs, config, shuffleData=False, augmentations=makeAugmentations())
    gen = createDataGenerator(dataDirs, config, shuffleData=False, augmentations=None)
    gen.normalizeImage = False

    steps_per_epoch = 3

    for inputs, dd1, dd2, dd3 in tqdm(gen.batches(steps_per_epoch), total=steps_per_epoch):
        for img in inputs:
            img = np.uint8(img)
            imshow(img=img[..., ::-1])
            if cv2.waitKey() == 27:
                return


main()
