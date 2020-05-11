import numpy as np
import cv2

from utils.MultiDirectoryBatchGenerator import _create_empty_xy, _encode_box, _find_match_anchor, _assign_box
from yolo.utils.box import create_anchor_boxes


def getDatasetItems(annotationObjects):
    xs = []
    ys_1 = []
    ys_2 = []
    ys_3 = []
    for annotationObj in annotationObjects:
        x, y1, y2, y3 = getDatasetItem(annotationObj)
        xs.append(x)
        ys_1.append(y1)
        ys_2.append(y2)
        ys_3.append(y3)

    return np.float32(xs), np.float32(ys_1), np.float32(ys_2), np.float32(ys_3)


def getDatasetItem(annotationObj):
    image, boxes, coded_labels = annotationObj
    # image = image / 255.

    list_ys = _create_empty_xy(image_size, numOfLabels)
    for original_box, label in zip(boxes, coded_labels):
        max_anchor, scale_index, box_index = _find_match_anchor(original_box, anchors)

        _coded_box = _encode_box(list_ys[scale_index], original_box, max_anchor, image_size, image_size)
        _assign_box(list_ys[scale_index], box_index, _coded_box, label)

    return image, list_ys[2], list_ys[1], list_ys[0]


anchors = create_anchor_boxes([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326])
labels = [*range(10)]
numOfLabels = len(labels)
net_size = 288
image_size = net_size


def makeImage(value):
    return np.full([image_size, image_size, 3], value, np.uint8)


def main():
    annObjects = [
        (makeImage(1), np.array([(10, 10, 20, 20)]), [1]),
        (makeImage(2), np.array([(10, 10, 20, 20), (10, 10, 60, 60)]), [2, 3])
    ]
    r1 = getDatasetItems(annObjects)

    annObjects = [
        (makeImage(3), np.array([(10, 10, 20, 20)]), [1]),
        (makeImage(4), np.array([(10, 10, 20, 20), (10, 10, 60, 60)]), [2, 5]),
        (makeImage(5), np.array([(10, 10, 20, 20), (10, 10, 60, 60), (30, 30, 60, 60)]), [2, 7, 9])
    ]
    r2 = getDatasetItems(annObjects)
    r1, r2



main()
