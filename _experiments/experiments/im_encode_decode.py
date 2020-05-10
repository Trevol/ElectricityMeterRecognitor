import cv2
import numpy as np

def main():
    filename = '/hdd/Datasets/counters/2_from_phone/000000.jpg'
    with open(filename, 'rb') as f:
        memoryBuffer = np.asarray(bytearray(f.read()), np.uint8)

    img = cv2.imdecode(memoryBuffer, flags=cv2.IMREAD_UNCHANGED)
    r, memoryBuffer = cv2.imencode('.bmp', img)
    if not r:
        raise Exception('failure of cv2.imencode')
    return memoryBuffer


main()
