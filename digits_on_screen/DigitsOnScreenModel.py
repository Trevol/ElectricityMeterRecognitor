import cv2
import numpy as np

from utils.bbox_utils import imageByBox
from utils.imutils import binarizeSauvola, imSize, fill, imshowWait, imInvert
from yolo.frontend import YoloDetector
from yolo.net import Yolonet


class DigitsOnScreenModel:
    labelNames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
    net_size = 320

    def __init__(self, weights):
        self._yolonet = None
        self._weights = weights

    def yolonet(self):
        return self._initYolonet()

    def _initYolonet(self):
        if self._yolonet:
            return self._yolonet
        nClasses = len(self.labelNames)
        self._yolonet = Yolonet(n_classes=nClasses)
        if self._weights:
            if self._weights.endswith('.h5'):  # keras format
                self._yolonet.load_weights(self._weights)
            else:  # darknet format. skip_detect_layer=True because where is no detection with initial weights
                self._yolonet.load_darknet_params(self._weights, skip_detect_layer=True)
        return self._yolonet

    def detect(self, image, threshold=.5):
        detector = YoloDetector(self.yolonet(), self.anchors, net_size=self.net_size)
        return detector.detect(image, threshold)  # boxes, labels, probs

    class _io:
        pad = 40

        def preprocess(self, image):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray = cv2.medianBlur(gray, 3, gray)
            inverted = imInvert(gray, out=gray)
            binarized = binarizeSauvola(inverted, windowSize=41, k=.1)
            binarized = imInvert(binarized, out=binarized)
            binarized = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)
            # imshowWait(DEBUG=binarized)
            return binarized

        def postprocess(self, boxes):
            pad = self.pad
            boxes = [(x1 - pad, y1 - pad, x2 - pad, y2 - pad) for x1, y1, x2, y2 in boxes]
            return boxes

        def extendDetectionBox(self, proposedBox):
            x1, y1, x2, y2 = proposedBox
            pad = self.pad
            return x1 - pad, y1 - pad, x2 + pad, y2 + pad

    def detectDigits(self, image, proposedBox):
        io = self._io()
        viewImage = imageByBox(image, proposedBox)
        preprocessed = io.preprocess(viewImage)
        boxes, labels, probs = self.detect(preprocessed, .9)
        # TODO: sort result by box.x1 (from left to right)
        # boxes = io.postprocess(boxes)
        return viewImage, boxes, labels, probs

    @classmethod
    def createWithLastWeights(cls):
        weights = "./weights/weights_7_3.263.h5"
        return cls(weights)
