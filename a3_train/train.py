import utils.suppressTfWarnings
from a3_train.MultiDirectoryBatchGenerator import MultiDirectoryBatchGenerator
from yolo.train import train_fn
from yolo.config import ConfigParser


def createDataGenerator(dataDirs, config, shuffle):
    return MultiDirectoryBatchGenerator(dataDirs,
                                        labels=config._model_config["labels"],
                                        batch_size=config._train_config["batch_size"],
                                        anchors=config._model_config["anchors"],
                                        min_net_size=config._train_config["min_size"],
                                        max_net_size=config._train_config["max_size"],
                                        jitter=config._train_config["jitter"],
                                        shuffle=shuffle)


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

    train_generator = createDataGenerator(trainDataDirs, config, True)
    valid_generator = createDataGenerator(valDataDirs, config, False)
    train_fn(model,
             train_generator,
             valid_generator,
             learning_rate=.0001,
             save_dname=save_dname,
             num_epoches=30)


if __name__ == '__main__':
    main()
