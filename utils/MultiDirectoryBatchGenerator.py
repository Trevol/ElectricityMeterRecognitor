# -*- coding: utf-8 -*-
import math
import os
import glob
from itertools import cycle

import cv2
import numpy as np
from albumentations import Compose, BboxParams

from utils.iter_utils import batchItems, unzip
from yolo.dataset.augment import resize_image
from yolo.utils.box import create_anchor_boxes
from yolo.dataset.annotation import parse_annotation
from yolo import COCO_ANCHORS

from random import shuffle
import itertools

# ratio between network input's size and network output's size, 32 for YOLOv3
DOWNSAMPLE_RATIO = 32


class MultiDirectoryBatchGenerator(object):
    def __init__(self,
                 dataDirectories,
                 labelNames,
                 batch_size,
                 anchors=COCO_ANCHORS,
                 image_size=320,
                 shuffleData=True,
                 augmentations=None,
                 normalizeImage=True):
        self.annotationObjects = AnnotationObject.loadFromDirectories(dataDirectories, labelNames, shuffleData)
        self._numOfLabels = len(labelNames)
        self.image_size = image_size
        self.anchors = create_anchor_boxes(anchors)
        self.batch_size = batch_size
        self.augmentations = self._composeAugmentations(augmentations)
        self.normalizeImage = normalizeImage

    def datasetBatchesCount(self):
        return math.ceil(len(self.annotationObjects) / self.batch_size)

    def batches(self, nBatches=None):
        if not nBatches:
            nBatches = self.datasetBatchesCount()
        annotationObjects = cycle(self.annotationObjects)
        annObjBatches = batchItems(annotationObjects, self.batch_size, nBatches)
        return (self._getDatasetItems(annObjBatch) for annObjBatch in annObjBatches)

    def _getDatasetItems(self, annotationObjects):
        xs = []
        ys_1 = []
        ys_2 = []
        ys_3 = []
        for annotationObj in annotationObjects:
            x, y1, y2, y3 = self._getDatasetItem(annotationObj)
            xs.append(x)
            ys_1.append(y1)
            ys_2.append(y2)
            ys_3.append(y3)

        return np.float32(xs), np.float32(ys_1), np.float32(ys_2), np.float32(ys_3)

    def _getDatasetItem(self, annotationObject):
        imageFile, boxes, coded_labels = annotationObject.annotationData()

        img = cv2.imread(imageFile)
        img, boxes, coded_labels = self._preprocessInputs(img, boxes, coded_labels)

        list_ys = _create_empty_xy(self.image_size, self._numOfLabels)
        for original_box, label in zip(boxes, coded_labels):
            max_anchor, scale_index, box_index = _find_match_anchor(original_box, self.anchors)

            _coded_box = _encode_box(list_ys[scale_index], original_box, max_anchor, self.image_size, self.image_size)
            _assign_box(list_ys[scale_index], box_index, _coded_box, label)

        return img, list_ys[2], list_ys[1], list_ys[0]

    def _preprocessInputs(self, img, boxes, labels):
        img, boxes, labels = self._augment(img, boxes, labels)
        # resize image and boxes, convert BGR to RGB
        img, boxes = resize_image(img, boxes, self.image_size, self.image_size)
        if self.normalizeImage:
            img = _normalize(img)
        return img, boxes, labels

    def _augment(self, image, boxes, labels):
        r = self.augmentations(image=image, bboxes=boxes, labels=labels)
        return r['image'], r['bboxes'], r['labels']

    @staticmethod
    def _composeAugmentations(augmentations):
        bbox_params = BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=.8)
        return Compose(augmentations or [], bbox_params=bbox_params)


class AnnotationObject:
    def __init__(self, annFile, labelNames):
        self._annFile = annFile
        self.labelNames = labelNames
        self._imageFile = None
        self._boxes = None
        self._labels = None
        self._initialized = False

    def annotationData(self):
        self._initialize()
        return self._imageFile, self._boxes, self._labels

    def _initialize(self):
        if self._initialized:
            return
        itemDir = os.path.split(self._annFile)[0]
        self._imageFile, self._boxes, self._labels = self._parseAnnotation(self._annFile, self.labelNames)
        self._initialized = True

    @staticmethod
    def _parseAnnotation(annFile, labelNames):
        itemDir = os.path.split(annFile)[0]
        imageFile, boxes, labels = parse_annotation(annFile, itemDir, labelNames)
        # preserve only desired objects (boxes with required labels)
        boxes, labels = unzip((b, l) for b, l in zip(boxes, labels) if l > -1)
        return imageFile, boxes, labels

    @classmethod
    def loadFromDirectories(cls, dataDirs, labelNames, shuffleData):
        annotationFiles = cls._loadAnnotationFiles(dataDirs, shuffleData)
        return [cls(f, labelNames) for f in annotationFiles]

    @staticmethod
    def _loadAnnotationFiles(dataDirs, shuffleData):
        filesByDir = (glob.glob(os.path.join(dataDir, '*.xml')) for dataDir in dataDirs)
        annotationFiles = itertools.chain.from_iterable(filesByDir)
        if shuffleData:
            annotationFiles = list(annotationFiles)
            shuffle(annotationFiles)
            return annotationFiles
        else:
            return sorted(annotationFiles)


def _create_empty_xy(net_size, n_classes, n_boxes=3):
    # get image input size, change every 10 batches
    base_grid_h, base_grid_w = net_size // DOWNSAMPLE_RATIO, net_size // DOWNSAMPLE_RATIO

    # initialize the inputs and the outputs
    ys_1 = np.zeros((1 * base_grid_h, 1 * base_grid_w, n_boxes, 4 + 1 + n_classes))  # desired network output 1
    ys_2 = np.zeros((2 * base_grid_h, 2 * base_grid_w, n_boxes, 4 + 1 + n_classes))  # desired network output 2
    ys_3 = np.zeros((4 * base_grid_h, 4 * base_grid_w, n_boxes, 4 + 1 + n_classes))  # desired network output 3
    list_ys = [ys_3, ys_2, ys_1]
    return list_ys


def _encode_box(yolo, original_box, anchor_box, net_w, net_h):
    x1, y1, x2, y2 = original_box
    _, _, anchor_w, anchor_h = anchor_box

    # determine the yolo to be responsible for this bounding box
    grid_h, grid_w = yolo.shape[:2]

    # determine the position of the bounding box on the grid
    center_x = .5 * (x1 + x2)
    center_x = center_x / float(net_w) * grid_w  # sigma(t_x) + c_x
    center_y = .5 * (y1 + y2)
    center_y = center_y / float(net_h) * grid_h  # sigma(t_y) + c_y

    # determine the sizes of the bounding box
    w = np.log(max((x2 - x1), 1) / float(anchor_w))  # t_w
    h = np.log(max((y2 - y1), 1) / float(anchor_h))  # t_h
    # print("x1, y1, x2, y2", x1, y1, x2, y2)

    box = [center_x, center_y, w, h]
    return box


def _find_match_anchor(box, anchor_boxes):
    """
    # Args
        box : array, shape of (4,)
        anchor_boxes : array, shape of (9, 4)
    """
    from yolo.utils.box import find_match_box
    x1, y1, x2, y2 = box
    shifted_box = np.array([0, 0, x2 - x1, y2 - y1])

    max_index = find_match_box(shifted_box, anchor_boxes)
    max_anchor = anchor_boxes[max_index]

    scale_index = max_index // 3
    box_index = max_index % 3
    return max_anchor, scale_index, box_index


def _assign_box(yolo, box_index, box, label):
    center_x, center_y, _, _ = box

    # determine the location of the cell responsible for this object
    grid_x = int(np.floor(center_x))
    grid_y = int(np.floor(center_y))

    # assign ground truth x, y, w, h, confidence and class probs to y_batch
    yolo[grid_y, grid_x, box_index] = 0
    yolo[grid_y, grid_x, box_index, 0:4] = box
    yolo[grid_y, grid_x, box_index, 4] = 1.
    yolo[grid_y, grid_x, box_index, 5 + label] = 1


def _normalize(image):
    return image / 255.


if __name__ == '__main__':
    def test():
        from tqdm import tqdm
        import utils.suppressTfWarnings
        from utils import imshow
        from yolo.config import ConfigParser
        import numpy as np
        import cv2
        import a3_train.augmentations as augmentations

        def createGenerator(dataDirs, config, shuffleData, augmentations):
            return MultiDirectoryBatchGenerator(dataDirs,
                                                labelNames=config._model_config["labels"],
                                                batch_size=config._train_config["batch_size"],
                                                anchors=config._model_config["anchors"],
                                                image_size=config._model_config["net_size"],
                                                shuffleData=shuffleData,
                                                augmentations=augmentations)

        dataDirs = [
            # '/hdd/Datasets/counters/0_from_internet/train',
            '/hdd/Datasets/counters/1_from_phone/train',
            '/hdd/Datasets/counters/2_from_phone/train'
        ]
        config = ConfigParser("configs/counters_screens.json")
        gen = createGenerator(dataDirs, config, shuffleData=False, augmentations=augmentations.make())
        # gen = createDataGenerator(dataDirs, config, shuffleData=False, augmentations=None)
        gen.normalizeImage = False
        gen.batch_size = 2

        steps_per_epoch = 111

        batches = gen.batches(steps_per_epoch)
        for inputs, dd1, dd2, dd3 in tqdm(batches, total=steps_per_epoch):
            for img in inputs:
                img = np.uint8(img)
                imshow(img=img[..., ::-1])
                if cv2.waitKey() == 27:
                    return


    test()
