"""
Title: Overlay
Date: 1/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
Description: Add information overlay on top of video output
"""

import sys
from pathlib import Path
import logging
import cv2
import numpy as np

# -------------

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
sys.path.insert(0, str(Path(SRC, 'lib')))  # ./lib
# print(sys.path[0])
sys.path.insert(0, str(Path(Path(__file__).parent, 'module')))

#--------------------Logger-------------------------#
FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
DATEFMT = '%d-%b-%y %H:%M:%S'

# logging.basicConfig(filename='test.log',filemode='w',format=FORMAT, level=logging.DEBUG)
logging.basicConfig(format=FORMAT, level=logging.INFO,
                    stream=sys.stdout, datefmt=DATEFMT)

log = logging.getLogger()

#----------------------------------------------------#


def information_position(frame, text):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CALCULATE INFORMATION OVERLAY POSITION
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    frame_width = frame.shape[0]
    # frame_heigth = frame.shape[1]
    text_length = len(text)
    y_position = 20
    x_position = frame_width - (10 * text_length)
    return x_position, y_position


def component_overlay(frame, text_true, text_none, color, position, display_flag=False):
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 0.6
    thickness = 2

    # display FPS
    if text_true is not None:
        cv2.putText(frame, text_true, position, font, fontScale,
                    color, thickness, lineType=16)
    else:
        cv2.putText(frame, text_none, position, font, fontScale,
                    color, thickness, lineType=16)


def rectangle_alert(frame, color):
    cv2.rectangle(frame, (0, 0), int(
        frame.shape[0], frame.shape[1]), color, 2, lineType=16)


def draw_overlay(frame, framerate, detection, framerate_color=(102, 0, 204), detection_color=(102, 0, 204), framerate_display_flag=True, rectangle_alert_flag=False):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # MAIN OVERLAY FUNCTION
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # display fps
    framerate_true = "FPS: {:.1f}".format(framerate)
    framerate_none = "FPS: 0"
    component_overlay(frame, framerate_true, framerate_none,
                      framerate_color, (10, 20), framerate_display_flag)

    # display detections
    detection_true = "Detection: {}".format(detection)
    detection_none = ""
    detection_position = information_position(frame, detection_true)
    component_overlay(frame, detection_true, detection_none,
                      detection_color, int(detection_position))

    # displaay rectangle alert
    if rectangle_alert_flag:
        rectangle_alert(frame, detection_color)

    return frame
