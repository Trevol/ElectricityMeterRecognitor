from albumentations import IAAPerspective
import numpy as np

from utils import imshowWait


def main():
    a = IAAPerspective(keep_size=False, always_apply=True)
    img = np.zeros([100, 100])
    img[25:75, 25:75] = 255
    key = 0
    while key != 27:
        transformedImg = a(image=img)['image']
        key = imshowWait(img=img, transformedImg=(transformedImg, transformedImg.shape))


main()
