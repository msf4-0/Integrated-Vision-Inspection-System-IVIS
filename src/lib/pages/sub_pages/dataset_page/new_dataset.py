"""
Title: New Dataset Page
Date: 7/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
import psycopg2
import pandas as pd
import numpy as np  # TEMP for table viz
from enum import IntEnum
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state as session_state

# DEFINE Web APP page configuration
layout = 'wide'
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)


# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
# TEST_MODULE_PATH = SRC / "test" / "test_page" / "module"

for path in sys.path:
    if str(LIB_PATH) not in sys.path:
        sys.path.insert(0, str(LIB_PATH))  # ./lib
    else:
        pass

    # if str(TEST_MODULE_PATH) not in sys.path:
    #     sys.path.insert(0, str(TEST_MODULE_PATH))
    # else:
    #     pass

from path_desc import chdir_root
from core.utils.code_generator import get_random_string
from core.utils.log import log_info, log_error  # logger
from core.webcam import webcam_webrtc
from data_manager.database_manager import init_connection, db_fetchone
from data_manager.dataset_management import NewDataset
from streamlit_webrtc import (
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])

# >>>> Variable Declaration
new_dataset = {}  # store
place = {}
DEPLOYMENT_TYPE = ("", "Image Classification", "Object Detection with Bounding Boxes",
                   "Semantic Segmentation with Polygons", "Semantic Segmentation with Masks")


class Random(IntEnum):
    Image_Classification = 1
    BBox = 2
    Polygons = 3
    Masks = 4


# >>>> Setup WebRTC >>>>
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)


def launch_webcam():
    """ Simple video loopback """
    webrtc_streamer(
        key="loopback",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=None,  # NoOp
    )


def show():

    chdir_root()  # change to root directory

    # >>>> START >>>>
    with st.sidebar.beta_container():

        st.image("resources/MSF-logo.gif", use_column_width=True)
    # with st.beta_container():
        st.title("Integrated Vision Inspection System", anchor='title')

        st.header(
            "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
        st.markdown("""___""")

    # <<<< START <<<<

    if "current_page" not in session_state:
        session_state.previous_page = "All Datasets"

    if "new_dataset" not in session_state:
        session_state.new_dataset = NewDataset(get_random_string(length=8))
        # SessionState.new_dataset["id"] = get_random_string(length=8)
        # set random dataset ID before getting actual from Database
    st.write(session_state.new_dataset.dataset_id)
    # >>>> Dataset Sidebar >>>>
    project_page_options = ("All Datasets", "New Dataset")
    with st.sidebar.beta_expander("Dataset Page", expanded=True):
        session_state.current_page = st.radio("project_page_select", options=project_page_options,
                                              index=0)
    # <<<< Dataset Sidebar <<<<

    # >>>> New Dataset MAIN >>>>
    # Page title
    st.write("# __Add New Dataset__")
    st.markdown("___")

    # right-align the dataset ID relative to the page
    id_blank, id_right = st.beta_columns([3, 1])
    id_right.write(
        f"### __Dataset ID:__ {session_state.new_dataset.dataset_id}")

    create_project_place = st.empty()
    # if layout == 'wide':
    #     col1, col2, col3 = create_project_place.beta_columns([1, 3, 1])
    # else:
    #     col2 = create_project_place
    with create_project_place.beta_container():
        st.write("## __Dataset Information :__")

        session_state.new_dataset.title = st.text_input(
            "Dataset Title", key="title", help="Enter the name of the dataset")
        place["title"] = st.empty()

        # **** Dataset Description (Optional) ****
        session_state.new_dataset.desc = st.text_area(
            "Description (Optional)", key="desc", help="Enter the description of the dataset")
        place["title"] = st.empty()

        session_state.new_dataset.deployment_type = st.selectbox(
            "Deployment Type", key="deployment_type", options=DEPLOYMENT_TYPE, format_func=lambda x: 'Select an option' if x == '' else x, help="Select the type of deployment of the dataset")
        place["deployment_type"] = st.empty()

        # >>>> Dataset Upload
        with st.beta_container():

            if 'webcam_flag' not in session_state:
                session_state.webcam_flag = False
                session_state.file_upload_flag = False

            def update_webcam_flag():
                session_state.webcam_flag = True
                session_state.file_upload_flag = False

            def update_file_uploader_flag():
                session_state.webcam_flag = False
                session_state.file_upload_flag = True
            # st.button()
            st.write("## __Dataset Upload:__")
            col1, col2, col3 = st.beta_columns([1, 1, 7])
            webcam_button = col1.button(
                "Webcam 📷", key="webcam_button", on_click=update_webcam_flag)
            file_upload_button = col2.button(
                "File Upload 📂", key="file_upload_button", on_click=update_file_uploader_flag)

            col1, col2 = st.beta_columns(2)
            if session_state.webcam_flag:
                with col1:
                    webcam_webrtc.app_loopback()
            elif session_state.file_upload_flag:
                uploaded_files_multi = col1.file_uploader(
                    label="Upload Image", type=['jpg', "png", "jpeg"], accept_multiple_files=True, key="upload")

        # **** Submit Button ****
        col1, col2 = st.beta_columns([3, 0.5])
        submit_button = col2.button("Submit", key="submit")

        st.write(vars(session_state.new_dataset))


def main():
    show()


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
