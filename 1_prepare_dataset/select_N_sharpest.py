import shutil

from imutils import paths
import cv2
import os
from glob import glob
import json
from itertools import groupby
import math

from utils import imshow, fit_image_to_shape, imageLaplacianSharpness
from utils.list_utils import splitList


def recreateDir(d):
    if os.path.isdir(d):
        # os.removedirs(d)
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=False)


def downsize_calcSharpness_saveDownsized(imagesBasePath, desiredSize, resizedImagesPath):
    recreateDir(resizedImagesPath)
    imagesWithSharpness = []
    for i, image_path in enumerate(sorted(paths.list_images(imagesBasePath))):
        image = cv2.imread(image_path)
        resizedImage = fit_image_to_shape(image, desiredSize)
        gray = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
        laplacianVariance = imageLaplacianSharpness(gray)

        imageFileName = f"{i:06d}" + os.path.splitext(image_path)[1]
        resizedImagePath = os.path.join(resizedImagesPath, imageFileName)
        imagesWithSharpness.append((resizedImagePath, laplacianVariance))
        cv2.imwrite(resizedImagePath, image)
    return imagesWithSharpness


def select_N_sharpest(imagesWithSharpness, N):
    def directory(imagesSharpnessItem):
        return os.path.split(imagesSharpnessItem[0])[0]

    totalFiles = len(imagesWithSharpness)
    dirs = ((dir, sorted(items)) for dir, items in groupby(imagesWithSharpness, directory))
    dirs = ((dir, items, len(items) / totalFiles) for dir, items in dirs)

    itemlen = lambda items, score: min(len(items), max(1, int(N * score)))
    dirs = ((dir, items, score, itemlen(items, score)) for dir, items, score in dirs)

    sharpestItems = []

    for dir, items, score, numOfParts in dirs:
        parts = splitList(items, numOfParts)
        sharpestItems.extend(max(p, key=lambda i: i[1]) for p in parts)

    return sharpestItems


def show(imagesWithSharpness):
    for i, (imagePath, sharpness) in enumerate(imagesWithSharpness):
        img = cv2.imread(imagePath)
        img = fit_image_to_shape(img, (1024, 1024))
        info = i, sharpness, os.path.basename(imagePath)
        imshow(img=(img, info))
        if cv2.waitKey() == 27:
            break


def split_train_val(images, trainRatio, baseDir):
    from random import shuffle
    shuffle(images)
    recreateDir(baseDir)


def steps():
    # downsize, calc sharpness and save downsized images to all_downsized
    imagesBasePath = '/hdd/Datasets/counters/1_from_phone/all'
    resizedImagesPath = '/hdd/Datasets/counters/1_from_phone/all_downsized'
    imagesWithSharpness: list = downsize_calcSharpness_saveDownsized(imagesBasePath, (1024, 1024), resizedImagesPath)

    # select N sharpest images (downsized)
    sharpestImages = select_N_sharpest(imagesWithSharpness, N=50)
    sharpestImages = [i for i, _ in sharpestImages]
    selectedImagesPath = '/hdd/Datasets/counters/1_from_phone/selected'
    recreateDir(selectedImagesPath)
    for imagePath in sharpestImages:
        shutil.copy(imagePath, selectedImagesPath)



steps()

# select_N_less_blurred()
# save_blurryness()
