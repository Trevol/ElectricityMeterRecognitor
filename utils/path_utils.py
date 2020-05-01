import os
import shutil


def recreateDir(d):
    if os.path.isdir(d):
        # os.removedirs(d)
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=False)
