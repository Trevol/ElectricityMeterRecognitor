import cv2
from utils import imshow


def noopCallback(event, x, y, flags, userData):
    pass


def getImage():
    return cv2.imread("../counter_images/01305.png", cv2.IMREAD_GRAYSCALE)


winName = "image"


def imageView(image):
    green = (0, 200, 0)

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
                cv2.circle(stateImage, pt2, 2, green, -1)
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
        showSelectionState()

    def cleanup():
        nonlocal state
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
