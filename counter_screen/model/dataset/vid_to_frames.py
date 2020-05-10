import cv2
import os

from utils.imutils import frames


def main():
    srcs = [
        '/hdd/Datasets/counters/0_from_phone/V_20200429_081246.mp4',
        '/hdd/Datasets/counters/0_from_phone/V_20200429_081205.mp4'
    ]
    for srcIndex, src in enumerate(srcs):
        dir, _ = os.path.splitext(src)
        if os.path.isdir(dir):
            os.removedirs(dir)
        os.makedirs(dir, exist_ok=False)
        for i, (frame, pos) in enumerate(frames(src)):
            if i > 0 and i % 100 == 0:
                print(i, src, dir)
            file = f'{srcIndex}{pos:06d}.jpg'
            cv2.imwrite(os.path.join(dir, file), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])


main()
