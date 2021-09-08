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
my_bar=st.progress(0)

x=[1,2,3,4,5]
y=[4,5,6,7,8]
z=5
list_of_groups=(x,y)
def test_find_file():
    with st.spinner(text='Finding in x'):
        if z in x:
            st.balloons()
            sleep(0.5)
            my_bar.progress(1/2)
    with st.spinner(text='Finding in y'):
        if z in y:
            st.balloons()

            sleep(0.5)
            my_bar.progress(2/2)

test_find_file()



    





# Store in dictionary
