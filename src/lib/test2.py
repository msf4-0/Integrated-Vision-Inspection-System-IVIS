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

from core.utils.log import log_info
path_to_archive = Path(
    "/home/rchuzh/Desktop/testing")
path_to_archive_dir=path_to_archive.parent
path_to_text = Path("/home/rchuzh/tf_list.txt")
st.write(path_to_archive_dir)

# @st.cache
# def load_file(path):
#     fp=open(path,'rb')
#     fp.seek(0)

#     return fp


# data = load_file(str(path_to_text))
def testing():
    print("This first")
    os.chdir("/home/rchuzh/Desktop")
    make_archive(base_name='some',
    format='zip',
    root_dir=path_to_archive_dir,
    base_dir=path_to_archive.relative_to(path_to_archive_dir))
st.button('archive',on_click=testing)

# with st.echo():
#     with open(path_to_archive,'rb') as fp:
#         print("Download section")
#         st.download_button('Download',
#                         data=fp,
#                         mime='application/zip',
#                         on_click=testing)
