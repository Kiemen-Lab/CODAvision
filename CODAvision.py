import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = "0"
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from gui.application import CODAVision

def main():
    CODAVision()

if __name__ == "__main__":
    main()