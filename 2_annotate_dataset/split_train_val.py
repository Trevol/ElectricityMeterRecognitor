import os
from glob import glob
from random import shuffle
import shutil

from yolo.dataset.annotation import parse_annotation


def main():
    allAnnotationDir = '/hdd/Datasets/counters/annotations'
    allImagesDir = '/hdd/Datasets/counters/img'
    trainDir = '/hdd/Datasets/counters/train'
    valDir = '/hdd/Datasets/counters/val'
    labels = ["counter", "counter_screen"]
    trainSplitRatio = .85

    allAnnotationsFiles = glob(os.path.join(allAnnotationDir, '*.xml'))

    # import cv2
    # for annotationFile in allAnnotationsFiles:
    #     imgFile, boxes, coded_labels = parse_annotation(annotationFile, allImagesDir, labels)
    #     assert cv2.imread(imgFile) is not None

    shuffle(allAnnotationsFiles)
    trainSplitLen = round(len(allAnnotationsFiles) * trainSplitRatio)
    trainAnnotationFiles = allAnnotationsFiles[:trainSplitLen]
    valAnnotationFiles = allAnnotationsFiles[trainSplitLen:]
    assert len(trainAnnotationFiles)
    assert len(valAnnotationFiles)

    copyImageAndAnnotations(trainAnnotationFiles, allImagesDir, trainDir)
    copyImageAndAnnotations(valAnnotationFiles, allImagesDir, valDir)


def copyImageAndAnnotations(annotationFiles, imagesDir, dstDir):
    if os.path.isdir(dstDir):
        os.removedirs(dstDir)
    os.makedirs(dstDir, exist_ok=False)
    for annotationFile in annotationFiles:
        imgFile, _, _ = parse_annotation(annotationFile, imagesDir, [])
        shutil.copy(imgFile, dstDir)
        shutil.copy(annotationFile, dstDir)


main()
