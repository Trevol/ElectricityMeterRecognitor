from functools import partial
from itertools import repeat
from typing import Tuple, List, Iterable, Generator

import numpy as np
import cv2
import os
from glob import glob
from random import sample, choices

from albumentations import BboxParams, Compose

from utils.iter_utils import batchItems, unzip
from utils.imutils import hStack


class NumberImageGenerator:
    k = 6
    hPad, vPad, middlePad = 48, 48, 10
    padding = hPad, vPad, middlePad

    def __init__(self, datasetDir, batchSize, augmentations):
        self.batchSize = batchSize
        self.augmentations = _utils.composeAugmentations(augmentations)
        self.numberImages = _utils.load(datasetDir)

    def batches(self, nBatches=None, DEBUG=False):
        digitsSampler = (choices(self.numberImages, k=self.k) for _ in repeat(None))

        numberImageFn = _utils.toAnnotatedNumberImage
        annotatedNumberImagesGen = (numberImageFn(labeledDigits, self.padding) for labeledDigits in digitsSampler)

        augmentFn = partial(_utils.augment, self.augmentations)
        augmentedImagesGen = ((annImg, augmentFn(annImg)) for annImg in annotatedNumberImagesGen)

        original_augmented_batches = batchItems(augmentedImagesGen, self.batchSize, nBatches or 100)

        yoloImagesBatch = [range(self.batchSize)]
        y1Batch = [range(self.batchSize)]
        y2Batch = [range(self.batchSize)]
        y3Batch = [range(self.batchSize)]
        yoloBatch = yoloImagesBatch, y1Batch, y2Batch, y3Batch
        imagesBatch_boxesBatch_labelsBatch_gen = ((yoloBatch, b) for b in original_augmented_batches)
        return imagesBatch_boxesBatch_labelsBatch_gen

    def __call__(self, nBatches=None):
        return self.batches(nBatches)


class _utils:
    @staticmethod
    def imread(path, invert):
        img = cv2.imread(path)
        if invert:
            img = np.subtract(255, img, out=img)
        return img

    @classmethod
    def load(cls, datasetDir):
        images = []
        for n in range(10):
            numFiles = os.path.join(datasetDir, f'numeric_{n}', '*.png')
            # todo: pad digit image after reading??
            numImages = [(n, cls.imread(f, True)) for f in glob(numFiles)]
            images.extend(numImages)
        return images

    @staticmethod
    def toAnnotatedNumberImage(labeledDigits, padding):
        """
        Combines labeled digits (list of (label, image)) to number image with digits annotations
        """
        labels, digits = unzip(labeledDigits)
        numberImage, boxes = hStack(digits, padding, fillValue=0)
        boxes = [(x1, y1 - 3, x2, y2 + 3) for x1, y1, x2, y2 in boxes]
        return numberImage, boxes, labels

    @staticmethod
    def composeAugmentations(augmentations):
        bbox_params = BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=.8)
        return Compose(augmentations or [], bbox_params=bbox_params)

    @staticmethod
    def augment(augmentations, image_boxes_labels):
        image, boxes, labels = image_boxes_labels
        r = augmentations(image=image, bboxes=boxes, labels=labels)
        return r['image'], r['bboxes'], r['labels']


class _yolo:
    DOWNSAMPLE_RATIO = 32

    @classmethod
    def format(cls, boxes, labels, net_size, nClasses, anchorBoxes):
        list_ys = cls.create_empty_xy(net_size, nClasses)
        for original_box, label in zip(boxes, labels):
            max_anchor, scale_index, box_index = cls.find_match_anchor(original_box, anchorBoxes)

            _coded_box = cls.encode_box(list_ys[scale_index], original_box, max_anchor, net_size, net_size)
            cls.assign_box(list_ys[scale_index], box_index, _coded_box, label)

        return list_ys[2], list_ys[1], list_ys[0]

    @classmethod
    def create_empty_xy(cls, net_size, n_classes, n_boxes=3):
        # get image input size, change every 10 batches
        base_grid_h = base_grid_w = net_size // cls.DOWNSAMPLE_RATIO

        # initialize the inputs and the outputs
        ys_1 = np.zeros((1 * base_grid_h, 1 * base_grid_w, n_boxes, 4 + 1 + n_classes))  # desired network output 1
        ys_2 = np.zeros((2 * base_grid_h, 2 * base_grid_w, n_boxes, 4 + 1 + n_classes))  # desired network output 2
        ys_3 = np.zeros((4 * base_grid_h, 4 * base_grid_w, n_boxes, 4 + 1 + n_classes))  # desired network output 3
        list_ys = [ys_3, ys_2, ys_1]
        return list_ys

    @staticmethod
    def encode_box(yolo, original_box, anchor_box, net_w, net_h):
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

    @staticmethod
    def find_match_anchor(box, anchor_boxes):
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

    @staticmethod
    def assign_box(yolo, box_index, box, label):
        center_x, center_y, _, _ = box

        # determine the location of the cell responsible for this object
        grid_x = int(np.floor(center_x))
        grid_y = int(np.floor(center_y))

        # assign ground truth x, y, w, h, confidence and class probs to y_batch
        yolo[grid_y, grid_x, box_index] = 0
        yolo[grid_y, grid_x, box_index, 0:4] = box
        yolo[grid_y, grid_x, box_index, 4] = 1.
        yolo[grid_y, grid_x, box_index, 5 + label] = 1
