# -*- coding: utf-8 -*-

import os
import glob

import cv2
import numpy as np
import tensorflow as tf
from albumentations import Compose, BboxParams

from yolo.dataset.augment import ImgAugment
from yolo.utils.box import create_anchor_boxes
from yolo.dataset.annotation import parse_annotation
from yolo import COCO_ANCHORS

from random import shuffle
import itertools

# ratio between network input's size and network output's size, 32 for YOLOv3
DOWNSAMPLE_RATIO = 32
DEFAULT_NETWORK_SIZE = 288


class MultiDirectoryBatchGenerator(object):
    def __init__(self,
                 dataDirectories,
                 labels,
                 batch_size,
                 anchors=COCO_ANCHORS,
                 min_net_size=320,
                 max_net_size=608,
                 shuffle=True,
                 augmentations=None,
                 steps_per_epoch=1000,
                 normalizeImage=True):

        self.ann_fnames = _getAnnotationFiles(dataDirectories)
        self.lable_names = labels
        self.min_net_size = (min_net_size // DOWNSAMPLE_RATIO) * DOWNSAMPLE_RATIO
        self.max_net_size = (max_net_size // DOWNSAMPLE_RATIO) * DOWNSAMPLE_RATIO
        self.anchors = create_anchor_boxes(anchors)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.augmentations = None
        if augmentations:
            self.augmentations = Compose(augmentations,
                                         bbox_params=BboxParams(format='pascal_voc', label_fields=['labels'],
                                                                min_visibility=.8))
        self.steps_per_epoch = steps_per_epoch
        self.normalizeImage = normalizeImage
        self._epoch = 0
        self._end_epoch = False
        self._index = 0
        self._initial_net_size()

    def _initial_net_size(self):
        self._net_size = DOWNSAMPLE_RATIO * (
                (self.min_net_size / DOWNSAMPLE_RATIO + self.max_net_size / DOWNSAMPLE_RATIO) // 2)
        self._net_size = int(self._net_size)
        print("_initial_net_size")
        print(self.min_net_size, self.max_net_size, self._net_size)

    def _update_net_size(self):
        self._net_size = DOWNSAMPLE_RATIO * np.random.randint(self.min_net_size / DOWNSAMPLE_RATIO, \
                                                              self.max_net_size / DOWNSAMPLE_RATIO + 1)

    def next_batch(self):
        if self._epoch >= 5:
            self._update_net_size()

        xs = []
        ys_1 = []
        ys_2 = []
        ys_3 = []
        for _ in range(self.batch_size):
            x, y1, y2, y3 = self._get()
            xs.append(x)
            ys_1.append(y1)
            ys_2.append(y2)
            ys_3.append(y3)

        if self._end_epoch == True:
            if self.shuffle:
                shuffle(self.ann_fnames)
            self._end_epoch = False
            self._epoch += 1

        return np.array(xs).astype(np.float32), np.array(ys_1).astype(np.float32), np.array(ys_2).astype(
            np.float32), np.array(ys_3).astype(np.float32)

    def _get(self):

        net_size = self._net_size

        # 1. get input file & its annotation
        annotationFileName = self.ann_fnames[self._index]
        directory = os.path.split(annotationFileName)[0]
        fname, boxes, coded_labels = parse_annotation(annotationFileName, directory, self.lable_names)

        # 2. read image in fixed size
        img_augmenter = ImgAugment(net_size, net_size, False)
        img, boxes_ = img_augmenter.imread(fname, boxes)
        # img, boxes, coded_labels = self._augment(cv2.imread(fname), boxes, coded_labels)

        # 3. Append ys
        list_ys = _create_empty_xy(net_size, len(self.lable_names))
        for original_box, label in zip(boxes, coded_labels):
            max_anchor, scale_index, box_index = _find_match_anchor(original_box, self.anchors)

            _coded_box = _encode_box(list_ys[scale_index], original_box, max_anchor, net_size, net_size)
            _assign_box(list_ys[scale_index], box_index, _coded_box, label)

        self._index += 1
        if self._index == self.steps_per_epoch:
            self._index = 0
            self._end_epoch = True
        if self.normalizeImage:
            img = _normalize(img)
        return img, list_ys[2], list_ys[1], list_ys[0]

    def _augment(self, image, boxes, labels):
        r = self.augmentations(image=image, bboxes=boxes, labels=labels)
        return r['image'], r['bboxes'], r['labels']


def _getAnnotationFiles(dataDirs):
    annotationFiles = []
    for dataDir in dataDirs:
        xmlFilesInDataDir = glob.glob(os.path.join(dataDir, "*.xml"))
        annotationFiles.extend(xmlFilesInDataDir)
    return annotationFiles

    # filesByDir = (glob.glob(os.path.join(dataDir, '*.xml')) for dataDir in dataDirs)
    # annotationFiles = itertools.chain.from_iterable(filesByDir)
    #
    # return sorted(annotationFiles)


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
