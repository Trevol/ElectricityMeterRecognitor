import cv2


def imshow(*unnamedMat, **namedMat):
    for i, mat in enumerate(unnamedMat):
        cv2.imshow(str(i), mat)
    for name, mat in namedMat.items():
        cv2.imshow(name, mat)
