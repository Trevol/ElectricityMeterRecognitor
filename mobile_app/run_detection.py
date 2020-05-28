import cv2
from counter_screen.model.CounterScreenModel import CounterScreenModel
from digits_on_screen.DigitsOnScreenModel import DigitsOnScreenModel
from utils import toInt
from utils.Timer import timeit
from utils.bbox_utils import imageByBox
from utils.detection_visualization import drawObjects
from utils.imutils import imshowWait


def frames():
    from glob import glob
    # imagesPattern = '/hdd/Datasets/counters/1_from_phone/1_all_downsized/*.jpg'
    # imagesPattern = '/hdd/Datasets/counters/2_from_phone/val/*.jpg'
    # imagesPattern = '/hdd/Datasets/counters/1_from_phone/val/*.jpg'
    # imagesPattern = '/hdd/Datasets/counters/3_from_phone/*.jpg'
    # imagesPattern = '/hdd/Datasets/counters/4_from_phone/*.jpg'
    imagesPattern = '/hdd/Datasets/counters/5_from_phone/*.jpg'
    for imagePath in sorted(glob(imagesPattern)):
        image = cv2.imread(imagePath)
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
        yield image, imagePath


def createScreenDetector():
    weights = '../counter_screen/model/weights/2_from_scratch/weights.h5'
    screenDetector = CounterScreenModel(weights)
    return screenDetector


def createDigitsDetector():
    weights = '../digits_on_screen/weights/weights_7_3.263.h5'
    digitsDetector = DigitsOnScreenModel(weights)
    return digitsDetector


def display(img, screenBox, digits, digitBoxes):
    x1, y1, x2, y2 = toInt(*screenBox)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 200), 1)

    return drawObjects(img, digitBoxes, digits, None)


def drawDigits(image, boxes, digits, color=(0, 200, 0)):
    for box, digit in zip(boxes, digits):
        x1, y1, x2, y2 = toInt(*box)

        textOrd = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(image, str(digit), textOrd, cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2)
    return image


def main():
    screenDetector = createScreenDetector()
    digitsDetector = createDigitsDetector()
    for image, imagePath in frames():
        with timeit('Detect screen'):
            screenBox = screenDetector.detectScreen(image)
        screenImg = None
        if screenBox is not None:
            with timeit('Detect digits on screen'):
                screenImg, digitBoxes, digits, digitProbs = digitsDetector.detectDigits(image, screenBox)
            # display(image, screenBox, digits, digitBoxes)
            drawDigits(screenImg, digitBoxes, digits)

        imgs = dict(
            img=(image[..., ::-1], imagePath),
            screenImg=screenImg[..., ::-1] if screenImg is not None else None
        )
        if imshowWait(**imgs) == 27:
            break


main()
