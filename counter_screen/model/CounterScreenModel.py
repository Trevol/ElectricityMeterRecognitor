from yolo.frontend import YoloDetector
from yolo.net import Yolonet


class CounterScreenModel:
    labelNames = ["counter", "counter_screen"]
    anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
    net_size = 416

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
