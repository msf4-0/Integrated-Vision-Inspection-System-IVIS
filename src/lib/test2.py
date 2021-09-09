"""
    TEST
    """
import sys
from pathlib import Path
from time import sleep
import cv2
import os
import numpy as np
import pandas as pd
import datetime
import psycopg2
from io import BytesIO
from shutil import make_archive
import streamlit as st
from PIL import Image
from streamlit import session_state as session_state
from streamlit.report_thread import add_report_ctx

SRC = Path(__file__).resolve().parents[1]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass
from training.labelmap_generator import labelmap_generator
from core.utils.log import log_info

labelmap_generator(framework='TensorFlow',
                   deployment_type='Object Detection with Bounding Boxes')


# Store in dictionary
