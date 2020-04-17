import cv2
import numpy as np

from utils import imshow


def main():
    img = cv2.imread("counter_images/01305.png")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (3, 3))

    cv2.calcHist()

    imshow(gray=gray, blur=blur)
    cv2.waitKey()


def calcHist(singleChannelImg):
    hist = cv2.calcHist([singleChannelImg], [0], None, [256], [0, 256])
    hist = np.squeeze(hist)
    return hist


def histTest():
    img = cv2.imread("../counter_images/01305.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur3 = cv2.blur(gray, (3, 3))
    blur5 = cv2.blur(gray, (5, 5))

    imshow(gray=gray, blur3=blur3, blur5=blur5)
    cv2.waitKey(1000)

    import matplotlib.pyplot as plt
    x = np.arange(256)

    # f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, sharex=True)
    # ax1.bar(x, calcHist(gray))
    # ax1.bar(x, calcHist(blur3))
    # ax1.bar(x, calcHist(blur5))

    f, ax = plt.subplots(1, 1)
    ax.bar(x, calcHist(gray), color='r', width=1)
    ax.bar(x + 1, calcHist(blur3), color='g', width=1)
    ax.bar(x + 2, calcHist(blur5), color='b', width=1)

    plt.show()


def edgesTest():
    img = cv2.imread("../counter_images/01305.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur3 = cv2.blur(gray, (3, 3))
    blur5 = cv2.blur(gray, (5, 5))

    imshow(gray=gray, blur3=blur3, blur5=blur5)

    incThresh1 = ord('q')
    decThresh1 = ord('a')
    incThresh2 = ord('e')
    decThresh2 = ord('d')
    esc = 27
    thresh1, thresh2 = 50, 150
    while True:
        print('thresh1, thresh2:', thresh1, thresh2)
        edges = cv2.Canny(blur5, thresh1, thresh2, L2gradient=True)
        imshow(edges=edges)

        key = cv2.waitKey(0)
        if key == esc:
            break
        elif key == incThresh1:
            thresh1 += 1
        elif key == decThresh1:
            thresh1 -= 1
        elif key == incThresh2:
            thresh2 += 1
        elif key == decThresh2:
            thresh2 -= 1


edgesTest()
