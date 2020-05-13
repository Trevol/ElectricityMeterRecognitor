from itertools import accumulate, zip_longest, repeat

import cv2
import numpy as np

from utils import imshow
from utils.iter_utils import unzip


def main():
    def op(arg):
        print('Op!!!', arg)
    True and op(1)
    False and op(2)
    None and op(1)

main()
