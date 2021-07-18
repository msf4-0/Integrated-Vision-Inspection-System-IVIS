"""
Title: Overlay
Date: 1/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
Description: Add information overlay on top of video output
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
for path in sys.path:
    if str(LIB_PATH) not in sys.path:
        sys.path.insert(0, str(LIB_PATH))  # ./lib
    else:
        pass


# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<


def information_position(frame, text):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CALCULATE INFORMATION OVERLAY POSITION
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    frame_width = frame.shape[0]
    # frame_heigth = frame.shape[1]
    text_length = len(text)
    y_position = 20
    x_position = frame_width - (10 * text_length)
    return int(x_position), int(y_position)


def component_overlay(frame, string, color, position, display_flag=False):
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 0.6
    thickness = 2

    # display FPS
    if string is not None:
        cv2.putText(frame, string, position, font, fontScale,
                    color, thickness, lineType=16)
    else:
        cv2.putText(frame, 'Null', position, font, fontScale,
                    color, thickness, lineType=16)


def rectangle_alert(frame, color):
    cv2.rectangle(frame, (0, 0), 
        (int(frame.shape[0]), int(frame.shape[1])), color, 2, lineType=16)


def draw_overlay(frame, framerate, detection, framerate_color=(102, 0, 204), detection_color=(102, 0, 204), framerate_display_flag=True, rectangle_alert_flag=False):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # MAIN OVERLAY FUNCTION
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # display fps
    if framerate is None:
        framerate_string = "FPS: 0"
    else:
        framerate_string = "FPS: {:.1f}".format(framerate)

    component_overlay(frame, framerate_string,
                      framerate_color, (10, 20), framerate_display_flag)

    # display detections
    if detection is None:
        detection_string = ""
    else:
        detection_string = "Detection: {}".format(detection)

    detection_position = information_position(frame, detection_string)
    component_overlay(frame, detection_string,
                      detection_color, detection_position)

    # displaay rectangle alert
    if rectangle_alert_flag:
        rectangle_alert(frame, detection_color)

    return frame
