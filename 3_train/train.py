# -*- coding: utf-8 -*-
def suppressTfWarnings():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '7'
    import warnings
    import logging
    # logging.getLogger('tensorflow').disabled = True
    warnings.filterwarnings('ignore')
    logging.disable(logging.WARNING)
    import tensorflow as tf


suppressTfWarnings()

from yolo.train import train_fn
from yolo.config import ConfigParser


def main():
    config = "configs/counters.json"
    config_parser = ConfigParser(config)

    train_generator, valid_generator = config_parser.create_generator()

    model = config_parser.create_model()

    learning_rate, save_dname, n_epoches = config_parser.get_train_params()

    train_fn(model,
             train_generator,
             valid_generator,
             learning_rate=learning_rate,
             save_dname=save_dname,
             num_epoches=n_epoches)


main()
