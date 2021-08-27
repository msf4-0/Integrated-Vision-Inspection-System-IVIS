"""
    TEST
    """
import sys
from base64 import b64encode
from io import BytesIO
from mimetypes import guess_type
from pathlib import Path
from threading import Thread
from timeit import timeit
from typing import Union

import cv2
import numpy as np
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

from core.utils.log import log_info


data_path = '/home/rchuzh/.local/share/integrated-vision-inspection-system/app_media/dataset/my-third-dataset/IMG_20210315_184149.jpg'
file_uri='file://'
href=file_uri+data_path

a_DOM=f'<a href={href} download>Download</a>'

st.markdown(a_DOM,unsafe_allow_html=True)