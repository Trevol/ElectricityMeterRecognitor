from itertools import accumulate, zip_longest, repeat

import cv2
import numpy as np

from utils import imshow
from utils.iter_utils import unzip


def main():
    m = list(repeat(True, -1))
    print(m)
    z = zip_longest([1, 2], [1], fillvalue=False)
    print(list(z))

main()
