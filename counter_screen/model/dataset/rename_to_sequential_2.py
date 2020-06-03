import os
from glob import glob
import shutil

import cv2
from xml.etree.ElementTree import parse

from yolo.dataset.annotation import PascalVocXmlParser


def read(path):
    with open(path, 'rb') as f:
        return f.read(-1)


def findAnnotationFile(imgPath, annotationDir):
    imgContent = read(imgPath)
    matchedImgFile = next((f for f in glob(os.path.join(annotationDir, '*.jpg')) if read(f) == imgContent), None)
    if matchedImgFile is None:
        return None
    dirAndBase, _ = os.path.splitext(matchedImgFile)
    return dirAndBase + ".xml"


def setNewPath(srcPath, newDir, newBaseName):
    _, ext = os.path.splitext(srcPath)
    newName = newBaseName + ext
    newPath = os.path.join(newDir, newName)
    os.rename(srcPath, newPath)
    return newPath


def showAnnotation(imagePath, annotationFile):
    p = PascalVocXmlParser(annotationFile)
    im = cv2.imread(imagePath)
    boxes = p.get_boxes()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 200, 00))
    cv2.imshow("", im)
    cv2.waitKeyEx()


def fixAnnotation(annFile):
    dir, xmlFileName = os.path.split(annFile)
    imgFileName = os.path.splitext(xmlFileName)[0] + '.jpg'
    imgDir = os.path.join(dir, imgFileName)

    tree = parse(annFile)
    root = tree.getroot()
    root.find("filename").text = imgFileName
    root.find("path").text = imgDir
    tree.write(annFile)


def main():
    path = '/hdd/Datasets/counters/Musson_counters/all/'
    annotationsDir = '/hdd/Datasets/counters/Musson_counters/all_annotated/'

    # for imgIndex, imgPath in enumerate(sorted(glob(os.path.join(path, '*.jpg')))):
    #     annotationFile = findAnnotationFile(imgPath, annotationsDir)
    #     if annotationFile is None or not os.path.isfile(annotationFile):
    #         raise Exception()
    #     imgDir = os.path.split(imgPath)[0]
    #     newImagePath = setNewPath(srcPath=imgPath, newDir=imgDir, newBaseName=f"{imgIndex:06d}")
    #     newAnnotationFile = setNewPath(srcPath=annotationFile, newDir=imgDir, newBaseName=f"{imgIndex:06d}")
    #     showAnnotation(newImagePath, newAnnotationFile)

    # for xmlPath in sorted(glob(os.path.join(path, '*.xml'))):
    #     fixAnnotation(xmlPath)


main()
