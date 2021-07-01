"""
Title: Preprocessor
Date: 30/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""


from pathlib import Path
from os import path
import sys
# -------------

# SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
# sys.path.insert(0, str(Path(SRC, 'lib')))  # ./lib
# # print(sys.path[0])
# sys.path.insert(0, str(Path(Path(__file__).parent, 'module')))
# --------------

# import streamlit as st
# from streamlit import cli as stcli
from time import perf_counter

import logging
import psycopg2
import cv2
from PIL import Image
import numpy as np


#--------------------Logger-------------------------#
FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
DATEFMT = '%d-%b-%y %H:%M:%S'

# logging.basicConfig(filename='test.log',filemode='w',format=FORMAT, level=logging.DEBUG)
logging.basicConfig(format=FORMAT, level=logging.INFO,
                    stream=sys.stdout, datefmt=DATEFMT)

log = logging.getLogger()

#----------------------------------------------------#


def resize_image(input, size):  # resize input image to model input size
    ih, iw = input.shape[:2]
    scale = min(size[1] / ih, size[0] / iw)
    resized_frame = cv2.resize(input, None, fx=scale, fy=scale)
    return resized_frame


def preprocess(self, inputs):
    """Preprocesses images for input into Execution Network
    Args:
        inputs (dictionary): input model data
    Returns:
        dictionary: processed input model data and frame metadata
    """

    image = inputs  # input image frame
    metadata = {}  # to store image dimensions
    # resize image to model input size
    resized_image = resize_image(
        image, (self.w, self.h))
    metadata = {'original_shape': image.shape,
                'resized_shape': resized_image.shape}

    # set h and w as height and width of resized_image
    h, w = resized_image.shape[:2]

    # check if h,w = model input h and w
    if h != self.h or w != self.w:
        # add constant 0 paddings to make resized dimension the same as model input
        # pad width format (Before, After) according to HWC
        resized_image = np.pad(resized_image, ((0, self.h - h), (0, self.w - w), (0, 0)),
                               mode='constant', constant_values=0)
    resized_image = resized_image.transpose(
        (2, 0, 1))  # Change data layout from HWC to CHW
    # reshape resized_image to correct shape -> n=1
    resized_image = resized_image.reshape((self.n, self.c, self.h, self.w))

    # store resized_image data into image_input dictionary
    image_input = {self.input_NCHW_blob_name: resized_image}
    # set h and w for NC input layout
    if self.input_NC_blob_name:
        image_input[self.input_NC_blob_name] = [self.h, self.w, 1]

    return image_input, metadata  # return image data and dimensions for scaling


def postprocess(self, raw_results, preprocess_metadata):
    """Scale bounding boxes for output frame
    Args:
        raw_results (dictionary): inference raw results
        preprocess_metadata (dictionary): preprocessing metadata
    Returns:
        dictionary: postprocessed results
    """

    detections = self.output_parser(raw_results)
    resized_image_shape = preprocess_metadata['resized_shape']
    orginal_image_shape = preprocess_metadata['original_shape']
    scale_x = self.w / resized_image_shape[1] * orginal_image_shape[1]
    scale_y = self.h / resized_image_shape[0] * orginal_image_shape[0]
    for detection in detections:
        detection.xmin *= scale_x
        detection.xmax *= scale_x
        detection.ymin *= scale_y
        detection.ymax *= scale_y
    return detections

