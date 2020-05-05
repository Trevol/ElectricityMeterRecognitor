import os
from glob import glob
from random import shuffle
import shutil

from utils.path_utils import recreateDir
from yolo.dataset.annotation import parse_annotation


def main():
    allAnnotationDir = '/hdd/Datasets/counters/2_from_phone/all'
    allImagesDir = allAnnotationDir
    trainDir = '/hdd/Datasets/counters/2_from_phone/train'
    valDir = '/hdd/Datasets/counters/2_from_phone/val'
    labels = ["counter", "counter_screen"]
    trainSplitRatio = .85

    allAnnotationsFiles = glob(os.path.join(allAnnotationDir, '*.xml'))

    shuffle(allAnnotationsFiles)
    trainSplitLen = round(len(allAnnotationsFiles) * trainSplitRatio)
    trainAnnotationFiles = allAnnotationsFiles[:trainSplitLen]
    valAnnotationFiles = allAnnotationsFiles[trainSplitLen:]
    assert len(trainAnnotationFiles)
    assert len(valAnnotationFiles)

    copyImageAndAnnotations(trainAnnotationFiles, allImagesDir, trainDir)
    copyImageAndAnnotations(valAnnotationFiles, allImagesDir, valDir)


def copyImageAndAnnotations(annotationFiles, imagesDir, dstDir):
    recreateDir(dstDir)
    for annotationFile in annotationFiles:
        imgFile, _, _ = parse_annotation(annotationFile, imagesDir, [])
        shutil.copy(imgFile, dstDir)
        shutil.copy(annotationFile, dstDir)


main()
