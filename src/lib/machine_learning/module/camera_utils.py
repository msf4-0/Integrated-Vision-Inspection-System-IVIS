"""
Title: Camera Utils
Date: 1/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
Description: Camera property utilities
"""

import sys
from pathlib import Path
import logging
from glob import iglob
import cv2
# import streamlit as st

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


def check_system_os():
    system_os = sys.platform

    return system_os


"""
if system_os.startswith('linux' || 'darwin'):
    camera_idx=[]
    camera_idx=return_camera_idx_linux()
elif system_os.startswith('win32' || 'cygwin'):
    camera_idx=[]
    camera_idx=return_camera_idx_windows()
"""


def return_camera_idx_linux():

    video_list = []
    video_open_list = []
    for camera in iglob("/dev/video?"):
        # c = cv2.VideoCapture(camera)
        # split '/dev/video0' -> '0'
        video_index = int(camera.split('/dev/video')[1])
        cap = cv2.VideoCapture(video_index)
        if cap.read()[0]:
            video_open_list.append(video_index)
            cap.release()
        # print(video_index,video_open_list)
        video_list.append(camera)
    if len(video_open_list) == 0:
        log.warning("No video devices found")
        st.error("No video devices found")
    video_list = sorted((video_list))
    video_open_list = sorted((video_open_list))

    return video_open_list


# print(return_camera_idx_linux())


def return_camera_idx_windows():
    from pymf import get_MF_devices  # install pymf.pyd
    device_list = get_MF_devices()
    for i, device_name in enumerate(device_list):
        print(f"opencv_index: {i}, device_name: {device_name}")


def dev_name(dev_path):
    from pyudev import Context
    for device in Context().list_devices(DEVNAME=dev_path):
        print(device.get('ID_MODEL_ENC'))
