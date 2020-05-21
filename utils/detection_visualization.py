from itertools import repeat

import cv2
import numpy as np

from utils import toInt
from utils.imutils import imWidth, imHeight


def drawObjects(image, boxes, labels, probs, color=(200, 0, 0), drawBoxes=True, drawLabels=True, drawProbs=True,
                printToConsole=True):
    if probs is None or len(probs) == 0:
        probs = repeat(1., len(boxes))
    if printToConsole:
        print("---------")
    for box, label, prob in zip(boxes, labels, probs):
        x1, y1, x2, y2 = toInt(*box)
        if drawBoxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

        if drawLabels:
            text = str(label)
            if drawProbs:
                probPct = int(round(prob * 100))
                if probPct < 100:
                    text = f"{label}:{probPct}"
            textOrd = x1, min(y2, imHeight(image)) - 1
            cv2.putText(image, text, textOrd, cv2.FONT_HERSHEY_SIMPLEX, .5, color)
        if printToConsole:
            print(label, prob)
    return image


def drawSeparateObjects(srcImage, boxes, labels, probs, drawProp=True, color=(0, 0, 200)):
    # stack from left to right
    images = []
    for box, label, prob in zip(boxes, labels, probs):
        image = srcImage.copy()
        images.append(image)
        x1, y1, x2, y2 = toInt(*box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        labelCoord = x1, y1

        labelText = str(label)
        if drawProp:
            scoreInPercents = int(round(prob * 100))
            if scoreInPercents < 100:
                labelText = f"{label}:{scoreInPercents}"

        cv2.putText(image, labelText, labelCoord, cv2.FONT_HERSHEY_SIMPLEX, .5, color)

    stack = np.vstack if imWidth(srcImage) > imHeight(srcImage) else np.hstack
    return stack(images)
