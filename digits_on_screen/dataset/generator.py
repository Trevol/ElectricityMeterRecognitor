from functools import partial
from itertools import repeat
from typing import Tuple, List, Iterable, Generator

import numpy as np
import cv2
import os
from glob import glob
from random import sample, choices, seed
from time import time_ns
from albumentations import BboxParams, Compose

from utils.iter_utils import batchItems, unzip
from utils.imutils import hStack, imSize, fill, imChannels
from yolo.dataset.augment import resize_image
from yolo.utils.box import create_anchor_boxes


class NumberImageGenerator:
    maxNumberOfDigits = 9
    hPad, vPad, middlePad = 48, 48, 10
    padding = hPad, vPad, middlePad
    nClasses = 10

    def __init__(self, datasetDir, batchSize, netSize, anchors, augmentations):
        self.batchSize = batchSize
        self.augmentations = _utils.composeAugmentations(augmentations)
        self.numberImages = _utils.load(datasetDir)
        self.netSize = netSize
        self._yolo = _yolo(netSize, self.nClasses, anchors)

    def datasetBatchesCount(self):
        return 100

    def batches(self, nBatches=None, DEBUG=False):
        def numOfDigits():
            return np.random.randint(1, self.maxNumberOfDigits + 1)

        digitsSampler = (choices(self.numberImages, k=numOfDigits()) for _ in repeat(None))

        numberImageFn = _utils.toAnnotatedNumberImage
        annotatedNumberImagesGen = (numberImageFn(labeledDigits, self.padding) for labeledDigits in digitsSampler)

        augmentFn = partial(_utils.augmentAndPadToNet, self.augmentations, self.netSize)
        augmentedImagesGen = ((annImg, augmentFn(annImg)) for annImg in annotatedNumberImagesGen)

        original_augmented_batches = batchItems(augmentedImagesGen, self.batchSize,
                                                nBatches or self.datasetBatchesCount())

        for b in original_augmented_batches:
            origBatch, augmentedBatch = unzip(b)
            yoloBatch = self._yolo.inputBatch(augmentedBatch)
            if DEBUG:
                yield yoloBatch, origBatch, augmentedBatch
            else:
                yield yoloBatch

        # imagesBatch_boxesBatch_labelsBatch_gen = \
        #     ((self._yolo.inputBatch(augmentedBatch), origBatch, augmentedBatch)
        #      for origBatch, augmentedBatch in original_augmented_batches)
        # return imagesBatch_boxesBatch_labelsBatch_gen

    def __call__(self, nBatches=None):
        return self.batches(nBatches)


class _utils:
    @staticmethod
    def imread(path, invert):
        img = cv2.imread(path)
        img = img[..., ::-1]
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
    def resizeImage(image, boxes, desired_w, desired_h):
        h, w = imSize(image)
        image = cv2.resize(image, (desired_h, desired_w))
        # fix object's position and size
        new_boxes = []
        kw = desired_w / w
        kh = desired_h / h
        for x1, y1, x2, y2 in boxes:
            x1 = round(x1 * kw)
            x1 = max(min(x1, desired_w), 0)
            x2 = round(x2 * kw)
            x2 = max(min(x2, desired_w), 0)

            y1 = round(y1 * kh)
            y1 = max(min(y1, desired_h), 0)
            y2 = round(y2 * kh)
            y2 = max(min(y2, desired_h), 0)
            new_boxes.append((x1, y1, x2, y2))
        return image, new_boxes

    @staticmethod
    def padToNetSize(image, netSize, fillValue=0):
        h, w = imSize(image)
        ch = imChannels(image)
        assert w <= netSize and h <= netSize
        leftSpacer = fill((h, netSize - w) + ch, fillValue)
        bottomSpacer = fill((netSize - h, netSize) + ch, fillValue)
        withLeftPad = np.hstack([image, leftSpacer])
        paddedToNet = np.vstack([withLeftPad, bottomSpacer])
        return paddedToNet

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
        bbox_params = BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=.5)
        return Compose(augmentations or [], bbox_params=bbox_params)

    @classmethod
    def augmentAndPadToNet(cls, augmentations, netSize, image_boxes_labels):
        image, boxes, labels = image_boxes_labels
        r = augmentations(image=image, bboxes=boxes, labels=labels)
        image, boxes, labels = r['image'], r['bboxes'], r['labels']
        image, boxes = cls.resizeImage(image, boxes, netSize, netSize)
        # image = cls.padToNetSize(image, netSize, fillValue=0)
        return image, boxes, labels


class _yolo:
    DOWNSAMPLE_RATIO = 32

    def __init__(self, net_size, nClasses, anchors):
        self.net_size = net_size
        self.nClasses = nClasses
        self.anchorBoxes = create_anchor_boxes(anchors)

    def input(self, image, boxes, labels):
        list_ys = self.create_empty_xy(self.net_size, self.nClasses)
        for original_box, label in zip(boxes, labels):
            max_anchor, scale_index, box_index = self.find_match_anchor(original_box, self.anchorBoxes)
            _coded_box = self.encode_box(list_ys[scale_index], original_box, max_anchor, self.net_size, self.net_size)
            self.assign_box(list_ys[scale_index], box_index, _coded_box, label)

        return np.divide(image, 255, dtype=np.float32), list_ys[2], list_ys[1], list_ys[0]

    def inputBatch(self, srcBatch):
        xs = []
        ys_1 = []
        ys_2 = []
        ys_3 = []
        for image, boxes, labels in srcBatch:
            x, y1, y2, y3 = self.input(image, boxes, labels)
            xs.append(x)
            ys_1.append(y1)
            ys_2.append(y2)
            ys_3.append(y3)
        return np.float32(xs), np.float32(ys_1), np.float32(ys_2), np.float32(ys_3)

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
