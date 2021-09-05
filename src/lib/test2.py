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
path_to_archive_dir = path_to_archive.parent
path_to_text = Path("/home/rchuzh/tf_list.txt")
st.write(path_to_archive_dir)


def testing():
    print("This first")
    os.chdir("/home/rchuzh/Desktop")
    make_archive(base_name='some',
                 format='zip',
                 root_dir=path_to_archive_dir,
                 base_dir=path_to_archive.relative_to(path_to_archive_dir))


st.button('archive', on_click=testing)

from typing import NamedTuple


if 'partition_size' not in session_state:
    session_state.partition_size = {'train': 0.8,
                                    'eval': 0.2,
                                    'test': 0
                                    }
if 'test_partition' not in session_state:
    session_state.test_partition = 0.1


def update_dataset_partition_ratio():

    # if session_state.test_partition == True:
    session_state.partition_size['train'] = session_state.partition_slider[0]
    session_state.partition_size['eval'] = round(session_state.partition_slider[1] -
                                                 session_state.partition_slider[0], 2)
    session_state.partition_size['test'] = round(
        1.0 - session_state.partition_slider[1], 2)


st.slider('Ratio', key='partition_slider',
             min_value=0.5,
             max_value=1.0,
             value=(0.8, 0.9),
             step=0.1,
             on_change=update_dataset_partition_ratio)

st.write(session_state.partition_slider)


st.info(f"""
### Train Dataset Ratio: {session_state.partition_size['train']}
### Evaluation Dataset Ratio: {session_state.partition_size['eval']}
### Test Dataset Ratio: {session_state.partition_size['test']}
""")

if session_state.partition_size['eval'] <= 0:
    st.error(f"Evaluation Dataset Partition Ratio should be more than 0.1")

st.write('Partition Size')
st.write(session_state.partition_size)


# Store in dictionary
