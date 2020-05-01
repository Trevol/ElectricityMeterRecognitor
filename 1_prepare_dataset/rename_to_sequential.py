
import os
from glob import glob
import shutil


def main():
    paths = [
        './*/*.*'        
    ]
    filePaths = (filePath for p in paths for filePath in glob(p))

    for i, filePath in enumerate(filePaths):
        directory, fileName = os.path.split(filePath)
        _, ext = os.path.splitext(fileName)
        if ext in ('.py', '.sh'):
            continue
        newName = f'{i:06d}{ext}'
        newFilePath = os.path.join(directory, newName)
        print(filePath, '  =>  ', newFilePath)
        os.rename(filePath, newFilePath)


main()