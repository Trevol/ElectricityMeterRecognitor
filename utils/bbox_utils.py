import cv2

from utils import toInt


def drawBoxes(image, boxes, color=(0, 0, 200)):
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = toInt(x1, y1, x2, y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
    return image


def imageByBox(srcImage, box, copy=True):
    x1, y1, x2, y2 = toInt(*box)
    boxImage = srcImage[y1:y2, x1:x2]
    if copy:
        boxImage = boxImage.copy()
    return boxImage
