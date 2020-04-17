import cv2
import numpy as np
from utils import imshow


def noopCallback(event, x, y, flags, userData):
    pass


def getImage():
    return cv2.imread("../counter_images/01305.png", cv2.IMREAD_GRAYSCALE)


winName = "image"


def imageView(image):
    green = (0, 200, 0)
    red = (0, 0, 200)
    elevationWndName = 'intensity-elevation'

    def showSelectionState(currentPoint=None):
        cv2.setWindowTitle(winName, state)
        stateImage = image.copy()
        if len(stateImage.shape) == 2:  # to BGR image
            stateImage = cv2.merge([stateImage, stateImage, stateImage])
        if state != 'initial':
            pt1 = selectedLine[0]
            pt2 = selectedLine[1]
            assert pt1
            cv2.circle(stateImage, pt1, 2, green, -1)
            # assert pt2 and currentPoint
            if pt2:
                cv2.circle(stateImage, pt2, 2, red, -1)
            pt2 = pt2 or currentPoint
            if pt2:
                cv2.line(stateImage, pt1, pt2 or currentPoint, green)

        cv2.imshow(winName, stateImage)

    def enterLineSelection(x, y):
        nonlocal state
        state = 'line-selection'
        selectedLine[0] = (x, y)
        showSelectionState()

    def lineSelected(x, y):
        nonlocal state
        state = 'line-selected'
        selectedLine[1] = (x, y)
        measureElevation()
        showSelectionState()

    def points(pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        xStep = (x2 - x1) // abs(x2 - x1)
        pts = []
        for x in range(x1, x2 + xStep, xStep):
            y = (y2 - y1) * (x - x1) / (x2 - x1) + y1
            pts.append((x, round(y)))
        return pts

    def measureElevation():
        pt1, pt2 = selectedLine
        assert pt1 and pt2
        plotHeight = 255
        elevationImage = np.zeros([plotHeight, image.shape[1]], np.uint8)
        assert len(image.shape) == 2  # for gray
        for x, y in points(pt1, pt2):
            xyIntensity = image[y, x]
            elevationImage[plotHeight - xyIntensity:, x] = max(xyIntensity, 40)

        cv2.imshow(elevationWndName, elevationImage)

    def cleanup():
        nonlocal state
        if state == 'line-selected':
            cv2.destroyWindow(elevationWndName)
        state = 'initial'
        selectedLine[0] = selectedLine[1] = None
        showSelectionState()

    def imageViewMouseHandler(event, x, y, flags, userData):
        if event == cv2.EVENT_LBUTTONUP:
            if state == 'initial' and flags & cv2.EVENT_FLAG_CTRLKEY:
                enterLineSelection(x, y)
            elif state == 'line-selection':
                lineSelected(x, y)
        if event == cv2.EVENT_MOUSEMOVE and state == 'line-selection':
            showSelectionState((x, y))

    selectedLine = [None, None]
    state = 'initial'

    cv2.namedWindow(winName)
    cv2.setMouseCallback(winName, imageViewMouseHandler)

    try:
        while True:
            showSelectionState()
            key = cv2.waitKey()
            if key == 27:
                if state == 'initial':
                    break
                else:
                    cleanup()
    finally:
        cv2.setMouseCallback(winName, noopCallback)


def main():
    image = getImage()
    imageView(image)


main()
