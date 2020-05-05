import os
import warnings
import logging


def suppressTfWarnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '7'
    # logging.getLogger('tensorflow').disabled = True
    warnings.filterwarnings('ignore')
    logging.disable(logging.WARNING)
    import tensorflow as tf


suppressTfWarnings()
