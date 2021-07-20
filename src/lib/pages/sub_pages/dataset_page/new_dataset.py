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
from copy import deepcopy
from time import sleep
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
from core.utils.file_handler import bytes_divisor
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])

# >>>> Variable Declaration >>>>
# new_dataset = {}  # store
place = {}
DEPLOYMENT_TYPE = ("", "Image Classification", "Object Detection with Bounding Boxes",
                   "Semantic Segmentation with Polygons", "Semantic Segmentation with Masks")


class Random(IntEnum):
    Image_Classification = 1
    BBox = 2
    Polygons = 3
    Masks = 4


def show():

    chdir_root()  # change to root directory

    with st.sidebar.beta_container():

        st.image("resources/MSF-logo.gif", use_column_width=True)
    # with st.beta_container():
        st.title("Integrated Vision Inspection System", anchor='title')

        st.header(
            "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
        st.markdown("""___""")

    # ******** SESSION STATE ********
    if "current_page" not in session_state:
        session_state.previous_page = "All Datasets"

    if "new_dataset" not in session_state:
        # set random dataset ID before getting actual from Database
        session_state.new_dataset = NewDataset(get_random_string(length=8))
        session_state.data_source = "File Upload 📂"
    # ******** SESSION STATE ********

    # >>>> Dataset SIDEBAR >>>>
    project_page_options = ("All Datasets", "New Dataset")
    with st.sidebar.beta_expander("Dataset Page", expanded=True):
        session_state.current_page = st.radio("project_page_select", options=project_page_options,
                                              index=0)
    # <<<< Dataset SIDEBAR <<<<

    # >>>>>>>> New Dataset INFO >>>>>>>>
    # Page title
    st.write("# __Add New Dataset__")
    st.markdown("___")

    # right-align the dataset ID relative to the page
    id_blank, id_right = st.beta_columns([3, 1])
    id_right.write(
        f"### __Dataset ID:__ {session_state.new_dataset.dataset_id}")

    create_dataset_place = st.empty()
    # if layout == 'wide':
    outercol1, outercol2, outercol3 = st.beta_columns([1.5, 3.5, 0.5])
    # else:
    #     outercol2 = st.beta_columns(1)

    # with st.beta_container():
    outercol1.write("## __Dataset Information :__")

    # >>>> CHECK IF NAME EXISTS CALLBACK >>>>
    def check_if_name_exist(field_placeholder, conn):
        context = ['name', session_state.name]
        if session_state.name:
            if session_state.new_dataset.check_if_exist(context, conn):
                field_placeholder['name'].error(
                    f"Dataset name used. Please enter a new name")
                sleep(1)
                log_error(f"Dataset name used. Please enter a new name")
            else:
                session_state.new_dataset.name = session_state.name
                log_error(f"Dataset name fresh and ready to rumble")

    outercol2.text_input(
        "Dataset Title", key="name", help="Enter the name of the dataset", on_change=check_if_name_exist, args=(place, conn,))
    place["name"] = outercol2.empty()

    # **** Dataset Description (Optional) ****
    description = outercol2.text_area(
        "Description (Optional)", key="desc", help="Enter the description of the dataset")
    if description:
        session_state.new_dataset.desc = description
    else:
        pass

    deployment_type = outercol2.selectbox(
        "Deployment Type", key="deployment_type", options=DEPLOYMENT_TYPE, format_func=lambda x: 'Select an option' if x == '' else x, help="Select the type of deployment of the dataset")

    if deployment_type is not None:
        session_state.new_dataset.deployment_type = deployment_type
        session_state.new_dataset.query_deployment_id()
        st.write(session_state.new_dataset.deployment_id)

    else:
        pass

    place["deployment_type"] = outercol2.empty()

    # <<<<<<<< New Dataset INFO <<<<<<<<

    # >>>>>>>> New Dataset Upload >>>>>>>>
    # with st.beta_container():

    # upload_dataset_place = st.empty()
    # if layout == 'wide':
    outercol1, outercol2, outercol3 = st.beta_columns([1.5, 3.5, 0.5])
    # else:
    # pass
    # if 'webcam_flag' not in session_state:
    #     session_state.webcam_flag = False
    #     session_state.file_upload_flag = False
    #     # session_state.img1=True

    outercol1.write("## __Dataset Upload:__")
    data_source_options = ["Webcam 📷", "File Upload 📂"]
    # col1, col2 = st.beta_columns(2)

    data_source = outercol2.radio(
        "Data Source", options=data_source_options, key="data_source_radio")
    data_source = data_source_options.index(data_source)

    outercol1, outercol2, outercol3 = st.beta_columns([1.5, 2, 2])
    dataset_size_string = f"- ### Number of datas: **{session_state.new_dataset.dataset_size}**"
    dataset_filesize_string = f"- ### Total size of data: **{(session_state.new_dataset.calc_total_filesize()):.2f} MB**"
    outercol3.markdown(" ____ ")

    dataset_size_place = outercol3.empty()
    dataset_size_place.write(dataset_size_string)

    dataset_filesize_place = outercol3.empty()
    dataset_filesize_place.write(dataset_filesize_string)

    outercol3.markdown(" ____ ")
    # TODO:KIV
    # >>>> WEBCAM >>>>
    if data_source == 0:
        with outercol2:
            webcam_webrtc.app_loopback()

    # >>>> FILE UPLOAD >>>>
    elif data_source == 1:
        uploaded_files_multi = outercol2.file_uploader(
            label="Upload Image", type=['jpg', "png", "jpeg"], accept_multiple_files=True, key="upload_widget")

        if uploaded_files_multi:
            session_state.new_dataset.dataset = deepcopy(uploaded_files_multi)

            session_state.new_dataset.dataset_size = len(
                uploaded_files_multi)  # length of uploaded files

            dataset_size_string = f"- ### Number of datas: **{session_state.new_dataset.dataset_size}**"
            dataset_filesize_string = f"- ### Total size of data: **{(session_state.new_dataset.calc_total_filesize()):.2f} MB**"

            outercol2.write(uploaded_files_multi[0])
            dataset_size_place.write(dataset_size_string)
            dataset_filesize_place.write(dataset_filesize_string)

    place["upload"] = outercol2.empty()
    # with st.beta_expander("Data Viewer", expanded=False):
    #     imgcol1, imgcol2, imgcol3 = st.beta_columns(3)
    #     imgcol1.checkbox("img1", key="img1")
    #     for image in uploaded_files_multi:
    #         imgcol1.image(uploaded_files_multi[1])

    # TODO: KIV

    # col1, col2, col3 = st.beta_columns([1, 1, 7])
    # webcam_button = col1.button(
    #     "Webcam 📷", key="webcam_button", on_click=update_webcam_flag)
    # file_upload_button = col2.button(
    #     "File Upload 📂", key="file_upload_button", on_click=update_file_uploader_flag)

    # <<<<<<<< New Dataset Upload <<<<<<<<
    # **** Submit Button ****
    success_place = st.empty()
    field = [session_state.new_dataset.name,
             session_state.new_dataset.deployment_id, session_state.new_dataset.dataset]
    st.write(field)
    submit_col1, submit_col2 = st.beta_columns([3, 0.5])
    submit_button = submit_col2.button("Submit", key="submit")

    if submit_button:
        session_state.new_dataset.has_submitted = session_state.new_dataset.check_if_field_empty(
            field, field_placeholder=place)

        if session_state.new_dataset.has_submitted:
            # TODO: Upload to database
            # st.success(""" Successfully created new dataset: {0}.
            #                 """.format(session_state.new_dataset.name))

            if session_state.new_dataset.save_dataset():

                success_place.success(
                    f"Successfully created **{session_state.new_dataset.name}** dataset")

                if session_state.new_dataset.insert_dataset():

                    success_place.success(
                        f"Successfully stored **{session_state.new_dataset.name}** dataset information in database")

                    # reset NewDataset class object
                    session_state.new_dataset = NewDataset(
                        get_random_string(length=8))

                else:
                    st.error(
                        f"Failed to stored **{session_state.new_dataset.name}** dataset information in database")
            else:
                st.error(
                    f"Failed to created **{session_state.new_dataset.name}** dataset")

    st.write(vars(session_state.new_dataset))
    # for img in session_state.new_dataset.dataset:
    #     st.image(img)


def main():
    show()s


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
