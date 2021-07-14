"""
Title: Webcam-WebRTC (Testing)
Date: 13/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
from typing import Optional
import psycopg2
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as SessionState
# NEW


# DEFINE Web APP page configuration
layout = 'wide'
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[3]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
for path in sys.path:
    if str(LIB_PATH) not in sys.path:
        sys.path.insert(0, str(LIB_PATH))  # ./lib
    else:
        pass
# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from core.utils.log import std_log  # logger
from data_manager.database_manager import init_connection

# @st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
# def init_connection():
#     return psycopg2.connect(**st.secrets["postgres"])

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration <<<<

# <<<< Variable Declaration <<<<


def show():

    chdir_root()  # change to root directory
    # initialise connection to Database
    conn = init_connection(**st.secrets["postgres"])
    with st.sidebar.beta_container():
        st.image("resources/MSF-logo.gif", use_column_width=True)

        st.title("Integrated Vision Inspection System", anchor='title')
        st.header(
            "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
        st.markdown("""___""")


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        show()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
