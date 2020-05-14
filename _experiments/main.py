from functools import partial
from itertools import accumulate, zip_longest, repeat

import cv2
import numpy as np

from utils.iter_utils import unzip


def main():
    def addFn(a, b):
        return a + b

    addBounded = partial(addFn, 2)
    print(addBounded(2))
    print(addBounded(3))
    print(addBounded(5.5))


main()
