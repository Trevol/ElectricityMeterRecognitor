from itertools import zip_longest, repeat
from typing import Union, Tuple
import cv2
import itertools
import numpy as np
from skimage.filters import threshold_sauvola


def imshow(*unnamedMat, **namedMat):
    for name, matOrMatWithTitle in itertools.chain(enumerate(unnamedMat), namedMat.items()):
        if matOrMatWithTitle is None:
            try:
                cv2.destroyWindow(name)
            except:
                pass
            continue
        if isinstance(matOrMatWithTitle, (tuple, list)) and len(matOrMatWithTitle) == 2:
            mat, title = matOrMatWithTitle
        else:
            mat, title = matOrMatWithTitle, None
        winName = str(name)
        cv2.imshow(winName, mat)
        if title is not None:
            cv2.setWindowTitle(winName, str(title))


def imshowWait(*unnamedMat, **namedMat):
    imshow(*unnamedMat, **namedMat)
    return cv2.waitKey()


def fit_image_to_shape(image, dstShape):
    dstH, dstW = dstShape
    imageH, imageW = image.shape[:2]

    scaleH = dstH / imageH
    scaleW = dstW / imageW
    scale = min(scaleH, scaleW)
    if scale >= 1:
        return image
    return cv2.resize(image, None, None, scale, scale)


def imResize(image, boxes, desired_w, desired_h):
    h, w = imSize(image)
    image = cv2.resize(image, (desired_h, desired_w))
    if boxes is None:
        return image, None
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


def fitToWidth(image, desiredWidth, fillValue):
    h, w = imSize(image)
    k = desiredWidth / w
    newHeight = int(h * k)
    newSize = desiredWidth, newHeight

    resized = cv2.resize(image, newSize)
    assert imSize(resized) == (newHeight, desiredWidth)

    bottomPad = fill((desiredWidth - newHeight, desiredWidth) + imChannels(image), fillValue)
    fittedImg = np.vstack([resized, bottomPad])
    return fittedImg


def imageLaplacianSharpness(image):
    # laplacian variance
    return cv2.Laplacian(image, cv2.CV_64F).var()


def frames(src: Union[int, str, cv2.VideoCapture], startPosition: int = 0, yieldPosition: bool = True):
    if isinstance(src, cv2.VideoCapture):
        src = src
        ownSrc = False
    else:
        src = cv2.VideoCapture(src)
        ownSrc = True
    try:
        src.set(cv2.CAP_PROP_POS_FRAMES, startPosition)
        pos = startPosition
        while True:
            ret, frame = src.read()
            if not ret:
                break
            item = (frame, pos) if yieldPosition else frame
            yield item
            pos += 1
    finally:
        if ownSrc:
            src.release()


def padImage(image, padding):
    pass


def full_like_channels(img, v, size):
    assert len(img.shape) in (2, 3)
    channelsTuple = img.shape[2:]  # will be empty if img is grayscale
    shape = tuple(size) + channelsTuple  # if a has depth - it will be added to size
    return np.full(shape, v, img.dtype)


def zeros_like_channels(img, size):
    assert len(img.shape) in (2, 3)
    channelsTuple = img.shape[2:]  # will be empty if img is grayscale
    shape = tuple(size) + channelsTuple  # if a has depth - it will be added to size
    return np.zeros(shape, img.dtype)


def imSize(img) -> Tuple[int, int]:
    return img.shape[:2]


def imHeight(img) -> int:
    return img.shape[0]


def imWidth(img) -> int:
    return img.shape[1]


def imChannels(img) -> Tuple:
    return img.shape[2:]


def zeros(shape, dtype=np.uint8):
    return np.zeros(shape, dtype)


def fill(shape, fillValue, dtype=np.uint8):
    return np.full(shape, fillValue, dtype)


def hStack(images, padding: Tuple[int, int, int], fillValue=0, defaultChannels=(3,)):
    h = imHeight(images[0]) if len(images) else 0
    channels = imChannels(images[0]) if len(images) else defaultChannels
    hPadding, vPadding, mPadding = padding

    hPadder = fill((h, hPadding) + channels, fillValue)
    mPadder = fill((h, mPadding) + channels, fillValue)
    boxes = []

    parts = [hPadder]
    x1, y1 = hPadding, vPadding
    for image, middlePad in zip_longest(images, repeat(True, len(images) - 1)):
        parts.append(image)
        x2, y2 = x1 + imWidth(image), y1 + imHeight(image)
        boxes.append((x1, y1, x2, y2))
        if middlePad:
            parts.append(mPadder)
            x1 += imWidth(image) + mPadding
    parts.append(hPadder)
    hStackedImage = np.hstack(parts)

    vPadder = fill((vPadding, imWidth(hStackedImage)) + channels, fillValue)

    resultImage = np.vstack([vPadder, hStackedImage, vPadder])
    return resultImage, boxes


def binarizeSauvola(gray, windowSize=41, k=.1):
    thresh_sauvola = threshold_sauvola(gray, window_size=windowSize, k=k)
    binary_sauvola = gray > thresh_sauvola
    binary_sauvola = np.uint8(binary_sauvola * 255)
    return binary_sauvola


def imInvert(image, out=None):
    return np.subtract(255, image, out=out)
