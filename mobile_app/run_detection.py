import cv2
from counter_screen.model.CounterScreenModel import CounterScreenModel
from digits_on_screen.DigitsOnScreenModel import DigitsOnScreenModel
from utils import toInt
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


def main():
    screenDetector = createScreenDetector()
    digitsDetector = createDigitsDetector()
    for image, imagePath in frames():
        box = screenDetector.detectScreen(image)
        if box is None:
            continue

        screenImg = imageByBox(image, box)
        digitBoxes, digits, digitProbs = digitsDetector.detectDigits(screenImg)

        # display(image, box, digits, digitBoxes)
        drawObjects(screenImg, digitBoxes, digits, digitProbs)

        imgs = dict(
            img=(image[..., ::-1], imagePath),
            screenImg=screenImg[..., ::-1]
        )
        if imshowWait(**imgs) == 27:
            break


main()
