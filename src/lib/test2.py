"""
    TEST
    """
import sys
from pathlib import Path
from time import sleep
import cv2
import numpy as np
import pandas as pd
import datetime
import psycopg2
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

random_place1 = st.empty()


with random_place1.container():

    st.write("Hi")
    st.write("Bye")
    st.text_input("sdadfafas")
    st.text_area("sfadfasdfadf")


sleep(2)
with random_place1.container():

    st.write("Change")
    st.write("Yeap")
    st.text_input("dfghdfghd", key="sfgsfgdfg")
    st.text_area("nfbncvncbvn", key='asfasdfadfadfadf')
    z = [{'ID': 1,
          'Task Name': 'IMG_20210316_082107.jpg',
          'Created By': '-',
          'Dataset Name': 'My Third Dataset',
          'Is Labelled': False,
          'Skipped': False,
          'Date/Time': datetime.datetime(2021, 8, 20, 18, 15, 13, 952586, tzinfo=psycopg2.tz.FixedOffsetTimezone(offset=480, name=None))},
         {'ID': 2,
          'Task Name': 'IMG_20210315_184229.jpg',
          'Created By': '-',
          'Dataset Name': 'My Third Dataset',
          'Is Labelled': False,
          'Skipped': False,
          'Date/Time': datetime.datetime(2021, 8, 20, 18, 15, 13, 955665, tzinfo=psycopg2.tz.FixedOffsetTimezone(offset=480, name=None))},
         {'ID': 3,
         'Task Name': 'IMG_20210315_184240.jpg',
          'Created By': '-',
          'Dataset Name': 'My Third Dataset',
          'Is Labelled': False,
          'Skipped': False,
          'Date/Time': datetime.datetime(2021, 8, 20, 18, 15, 13, 957486, tzinfo=psycopg2.tz.FixedOffsetTimezone(offset=480, name=None))}]

    df = pd.DataFrame(z)
    st.write(df)
