import os
import logging
from datetime import datetime


# Configure Python's standard logging
# Set DEBUG_APP_MODE to True to see DEBUG level logs from Python and OpenCV warnings.
# Set it to False for production/normal use to see only INFO and above from Python,
# and to suppress OpenCV warnings.
DEBUG_APP_MODE = False # Toggle this for debugging

APP_LOG_LEVEL = logging.DEBUG if DEBUG_APP_MODE else logging.INFO

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(logs_dir, exist_ok=True)
log_filename = os.path.join(logs_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Basic configuration for Python's logging
logging.basicConfig(
    level=APP_LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Console output
    ]
)
# Create a logger for the application
logger = logging.getLogger("CODAvision")

# Conditionally set OpenCV's C++ log level based on Python's logging level.
# This must be done BEFORE cv2 is imported anywhere by the application.
effective_python_log_level = logging.getLogger().getEffectiveLevel()

if effective_python_log_level <= logging.DEBUG:
    # If Python logging is set to DEBUG or more verbose, allow OpenCV warnings to be printed.
    os.environ['OPENCV_LOG_LEVEL'] = 'WARNING'
    # logger.debug("Python logging level is DEBUG. OpenCV log level set to WARNING (OpenCV warnings will be shown).")
else:
    # If Python logging is INFO or less verbose, suppress OpenCV warnings by setting its log level to ERROR.
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    # logger.info("Python logging level is INFO or higher. OpenCV log level set to ERROR (OpenCV warnings will be suppressed).")

# Ensure OPENCV_IO_MAX_IMAGE_PIXELS is set.
if 'OPENCV_IO_MAX_IMAGE_PIXELS' not in os.environ:
    os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = "0"
    logger.debug("OPENCV_IO_MAX_IMAGE_PIXELS set to '0'.")
elif os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] != "0":
    logger.warning(
        f"OPENCV_IO_MAX_IMAGE_PIXELS was already set to {os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']}. "
        "Overriding to '0' per CODAvision standard."
    )
    os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = "0"

from PIL import Image # PIL import after os.environ changes
Image.MAX_IMAGE_PIXELS = None

# Now import other modules from the application.
# These modules might import cv2, and the OPENCV_LOG_LEVEL setting will now be in effect.
from gui.application import CODAVision

def main():
    logger.info("Starting CODAvision application...")
    CODAVision()
    logger.info("CODAvision application finished.")

if __name__ == "__main__":
    main()