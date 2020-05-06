from albumentations import Compose, BboxParams

import utils.suppressTfWarnings
from a3_train.MultiDirectoryBatchGenerator import MultiDirectoryBatchGenerator
from yolo.train import train_fn
from yolo.config import ConfigParser
import a3_train.augmentations as augmentations

def createDataGenerator(dataDirs, config, shuffle, augmentations, steps_per_epoch):
    return MultiDirectoryBatchGenerator(dataDirs,
                                        labels=config._model_config["labels"],
                                        batch_size=config._train_config["batch_size"],
                                        anchors=config._model_config["anchors"],
                                        image_size=config._model_config["net_size"],
                                        shuffle=shuffle,
                                        augmentations=augmentations,
                                        steps_per_epoch=steps_per_epoch)


def makeAugmentations():
    return Compose(augmentations.make(), bbox_params=BboxParams(format='pascal_voc', min_visibility=.8))


def main():
    configFile = "configs/counters.json"
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

    learning_rate, save_dname, n_epoches = config.get_train_params()

    augmentations = makeAugmentations()
    train_generator = createDataGenerator(trainDataDirs, config, True, augmentations, 1000)
    valid_generator = createDataGenerator(valDataDirs, config, False, None, None)
    train_fn(model,
             train_generator,
             valid_generator,
             learning_rate=.0001,
             save_dname=save_dname,
             num_epoches=30)


if __name__ == '__main__':
    main()
