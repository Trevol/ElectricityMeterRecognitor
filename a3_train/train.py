from albumentations import Compose, BboxParams

import utils.suppressTfWarnings
from utils.MultiDirectoryBatchGenerator import MultiDirectoryBatchGenerator
from yolo.train import train_fn
from yolo.config import ConfigParser
import a3_train.augmentations as augmentations


def createDataGenerator(dataDirs, config, shuffleData, augmentations):
    return MultiDirectoryBatchGenerator(dataDirs,
                                        labelNames=config._model_config["labels"],
                                        batch_size=config._train_config["batch_size"],
                                        anchors=config._model_config["anchors"],
                                        image_size=config._model_config["net_size"],
                                        shuffleData=shuffleData,
                                        augmentations=augmentations)


def main():
    configFile = "configs/counters_screens.json"
    config = ConfigParser(configFile)

    trainDataDirs = [
        # '/hdd/Datasets/counters/0_from_internet/train',
        '/hdd/Datasets/counters/1_from_phone/train',
        '/hdd/Datasets/counters/2_from_phone/train'
    ]
    valDataDirs = [
        # '/hdd/Datasets/counters/0_from_internet/val',
        '/hdd/Datasets/counters/1_from_phone/val',
        '/hdd/Datasets/counters/2_from_phone/val'
    ]

    model = config.create_model()

    learning_rate, saveDir, n_epoches = config.get_train_params()

    augments = augmentations.make()
    train_generator = createDataGenerator(trainDataDirs, config, True, augments)
    valid_generator = createDataGenerator(valDataDirs, config, False, None)
    train_fn(model,
             train_generator,
             valid_generator,
             learning_rate=.0005,
             saveDir=saveDir,
             num_epoches=30,
             stepsPerEpoch=1000)


if __name__ == '__main__':
    main()
