import cv2
import matplotlib.pyplot as plt

from utils import fit_image_to_shape, imshow
from yolo.utils.box import visualize_boxes
from yolo.config import ConfigParser
import cv2
from glob import glob
import os


def main():
    imagesPattern = '/hdd/Datasets/counters/0_from_phone/V_20200429_081246/*.jpg'

    configFile = "configs/counters.json"

    config_parser = ConfigParser(configFile)
    model = config_parser.create_model(skip_detect_layer=False)
    detector = config_parser.create_detector(model)

    for image_path in sorted(glob(imagesPattern)):
        if os.path.splitext(image_path)[1] == '.xml':
            continue
        image = cv2.imread(image_path)
        image = image[..., ::-1]  # to RGB
        image = fit_image_to_shape(image, (1000, 1800))

        boxes, labels, probs = detector.detect(image, 0.5)

        labelNames = config_parser.get_labels()
        labelIndex = {_id: name for _id, name in enumerate(labelNames)}
        print(labels, [labelIndex[_id] for _id in labels])
        visualize_boxes(image, boxes, labels, probs, labelNames)

        imshow(img=image[..., ::-1])
        cv2.setWindowTitle("img", image_path)
        if cv2.waitKey() == 27:
            break


main()
