from typing import Union
import cv2
import itertools


def imshow(*unnamedMat, **namedMat):
    for name, matOrMatWithTitle in itertools.chain(enumerate(unnamedMat), namedMat.items()):
        if isinstance(matOrMatWithTitle, (tuple, list)) and len(matOrMatWithTitle) == 2:
            mat, title = matOrMatWithTitle
        else:
            mat, title = matOrMatWithTitle, None
        cv2.imshow(name, mat)
        if title is not None:
            cv2.setWindowTitle(name, str(title))


def fit_image_to_shape(image, dstShape):
    dstH, dstW = dstShape
    imageH, imageW = image.shape[:2]

    scaleH = dstH / imageH
    scaleW = dstW / imageW
    scale = min(scaleH, scaleW)
    if scale >= 1:
        return image
    return cv2.resize(image, None, None, scale, scale)


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
