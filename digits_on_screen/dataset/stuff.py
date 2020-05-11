from random import sample, choice, choices

import cv2
import glob
import pickle

from utils import imshow, imshowWait


def main():
    f = './28x28/numeric_?/*.png'
    files = sorted(glob.glob(f))

    images = [(cv2.imread(file), file) for file in files]
    for img in images:
        if imshowWait(img=img) == 27:
            break


def main():
    file = './28x28/28x28.pickle'
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding="bytes")
    images = data[b'images']
    while True:
        im = choice(images)
        if imshowWait(im) == 27: break


main()
