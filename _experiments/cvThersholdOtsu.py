import cv2
import numpy as np

from utils import imshow


def main():
    # image = cv2.imread("../counter_images/01305.png", cv2.IMREAD_GRAYSCALE)
    image = cv2.imread("../counter_images/01305.png")
    image = cv2.blur(image, (5, 5))
    # image = 255 - image

    # image = image[59:201, 103:187]
    saliency = cv2.saliency_StaticSaliencyFineGrained.create()
    ret, salImage = saliency.computeSaliency(image)

    salImage = np.uint8(salImage * 255 / salImage.max())

    imshow(image, salImage)

    cv2.waitKey()


main()
