from counter_screen.model.CounterScreenModel import CounterScreenModel
from utils.imutils import fit_image_to_shape, imshowWait
from yolo.utils.box import visualize_boxes
import cv2
from glob import glob
import os


def main():
    # imagesPattern = '/hdd/Datasets/counters/1_from_phone/1_all_downsized/*.jpg'
    # imagesPattern = '/hdd/Datasets/counters/2_from_phone/val/*.jpg'
    imagesPattern = '/hdd/Datasets/counters/1_from_phone/val/*.jpg'
    imagesPattern = '/hdd/Datasets/counters/3_from_phone/*.jpg'

    detector = CounterScreenModel('weights/2_from_scratch/weights.h5')

    for image_path in sorted(glob(imagesPattern)):
        if os.path.splitext(image_path)[1] == '.xml':
            continue
        image = cv2.imread(image_path)[..., ::-1]  # to RGB
        image = fit_image_to_shape(image, (1000, 1800))

        boxes, labels, probs = detector.detect(image, 0.5)

        labelNames = detector.labelNames
        labelIndex = {_id: name for _id, name in enumerate(labelNames)}
        print(labels, [labelIndex[_id] for _id in labels])
        visualize_boxes(image, boxes, labels, probs, labelNames)

        if imshowWait(img=(image[..., ::-1], image_path)) == 27:
            break


main()
