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
from streamlit_webrtc import (
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
# DEFINE Web APP page configuration
# layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

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
from core.utils.log import log_info  # logger
from data_manager.database_manager import init_connection

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Setup WebRTC >>>>
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)
# <<<< Setup WebRTC <<<<


def app_loopback():
    """ Simple video loopback """
    webrtc_streamer(
        key="loopback",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=WEBRTC_CLIENT_SETTINGS['rtc_configuration'],
        media_stream_constraints=WEBRTC_CLIENT_SETTINGS['media_stream_constraints'],
        video_processor_factory=None,  # NoOp
    )


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
    app_loopback()


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        show()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
