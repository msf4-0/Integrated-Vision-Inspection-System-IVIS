"""
Title: Existing Dataset Page
Date: 2/8/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np  # TEMP for table viz
from enum import IntEnum
from copy import deepcopy
from time import sleep
from typing import Dict, List, Union, Optional
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
from core.utils.file_handler import bytes_divisor
from core.webcam import webcam_webrtc
from data_manager.database_manager import init_connection, db_fetchone
from data_manager.dataset_management import Dataset, NewDataset, query_dataset_list, get_dataset_name_list
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])

# >>>> Variable Declaration >>>>
# new_dataset = {}  # store
place = {}
DEPLOYMENT_TYPE = ("", "Image Classification", "Object Detection with Bounding Boxes",
                   "Semantic Segmentation with Polygons", "Semantic Segmentation with Masks")


class DeploymentType(IntEnum):
    Image_Classification = 1
    OD = 2
    Instance = 3
    Semantic = 4

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return DeploymentType[s]
        except KeyError:
            raise ValueError()


def show(dataset_info: Dict = None):

    chdir_root()  # change to root directory

    # Instantiate Dataset Class
    dataset_id = dataset_info.ID
    dataset = Dataset(dataset_info)

    # ******** SESSION STATE ********
    if "current_page" not in session_state:
        session_state.previous_page = "All Datasets"

    if "dataset" not in session_state:
        # set random dataset ID before getting actual from Database
        session_state.data_source = "File Upload 📂"  # for Dataset upload

    # ******** SESSION STATE ********

    # >>>> Dataset SIDEBAR >>>>
    project_page_options = ("All Datasets", "New Dataset")
    with st.sidebar.beta_expander("Dataset Page", expanded=True):
        session_state.current_page = st.radio("", options=project_page_options,
                                              index=0)
    # <<<< Dataset SIDEBAR <<<<

# TODO #17 Load details of existing dataset

    # right-align the dataset ID relative to the page
    id_blank, id_right = st.beta_columns([3, 1])
    if dataset_info:
        id_right.write(
            f"### __Dataset ID:__ {dataset.id}")  # DATASET ID

    # NOTE *******************************
    outercol1, outercol2, outercol3 = st.beta_columns([1.5, 3.5, 0.5])

    # outercol1.write("## __Dataset Information :__")

    # >>>> CHECK IF NAME EXISTS CALLBACK >>>>

    def check_if_name_exist(field_placeholder, conn):
        context = ['name', session_state.name]
        if session_state.name:
            if dataset.check_if_exist(context, conn):
                field_placeholder['name'].error(
                    f"Dataset name used. Please enter a new name")
                sleep(1)
                log_error(f"Dataset name used. Please enter a new name")
            else:
                dataset.name = session_state.name
                log_error(f"Dataset name fresh and ready to rumble")

    outercol2.text_input(
        "Dataset Title", key="name", value=dataset.name, help="Enter the name of the dataset", on_change=check_if_name_exist, args=(place, conn,))
    place["name"] = outercol2.empty()

    # **** Dataset Description (Optional) ****
    description = outercol2.text_area(
        "Description (Optional)", value=dataset.desc if dataset.desc else "Describe your dataset", key="desc", help="Enter the description of the dataset")
    if description:
        dataset.desc = description
    else:
        pass

    deployment_type = outercol2.selectbox(
        "Deployment Type", key="deployment_type", index=DEPLOYMENT_TYPE.index(dataset.deployment_type), options=DEPLOYMENT_TYPE, format_func=lambda x: 'Select an option' if x == '' else x, help="Select the type of deployment of the dataset")

    if deployment_type is not None:
        dataset.deployment_type = deployment_type
        dataset.query_deployment_id()

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
    dataset_size_string = f"- ### Number of datas: **{dataset.dataset_size}**"
    dataset_filesize_string = f"- ### Total size of data: **{(dataset.calc_total_filesize()):.2f} MB**"
    outercol3.markdown(" ____ ")

    dataset_size_place = outercol3.empty()
    dataset_size_place.write(dataset_size_string)

    dataset_filesize_place = outercol3.empty()
    dataset_filesize_place.write(dataset_filesize_string)

    outercol3.markdown(" ____ ")
    # TODO: #15 Webcam integration
    # >>>> WEBCAM >>>>
    if data_source == 0:
        with outercol2:
            webcam_webrtc.app_loopback()

    # >>>> FILE UPLOAD >>>>
    elif data_source == 1:
        uploaded_files_multi = outercol2.file_uploader(
            label="Upload Image", type=['jpg', "png", "jpeg"], accept_multiple_files=True, key="upload_widget")

        if uploaded_files_multi:
            dataset.dataset = deepcopy(uploaded_files_multi)

            dataset.dataset_size = len(
                uploaded_files_multi)  # length of uploaded files

            dataset_size_string = f"- ### Number of datas: **{dataset.dataset_size}**"
            dataset_filesize_string = f"- ### Total size of data: **{(dataset.calc_total_filesize()):.2f} MB**"

            outercol2.write(uploaded_files_multi[0])
            dataset_size_place.write(dataset_size_string)
            dataset_filesize_place.write(dataset_filesize_string)

    place["upload"] = outercol2.empty()
    # with st.beta_expander("Data Viewer", expanded=False):
    #     imgcol1, imgcol2, imgcol3 = st.beta_columns(3)
    #     imgcol1.checkbox("img1", key="img1")
    #     for image in uploaded_files_multi:
    #         imgcol1.image(uploaded_files_multi[1])

    # TODO: #15

    # col1, col2, col3 = st.beta_columns([1, 1, 7])
    # webcam_button = col1.button(
    #     "Webcam 📷", key="webcam_button", on_click=update_webcam_flag)
    # file_upload_button = col2.button(
    #     "File Upload 📂", key="file_upload_button", on_click=update_file_uploader_flag)

    # <<<<<<<< New Dataset Upload <<<<<<<<
    # **** Submit Button ****
    success_place = st.empty()
    field = [dataset.name,
             dataset.deployment_id, dataset.dataset]
    st.write(field)
    submit_col1, submit_col2 = st.beta_columns([3, 0.5])
    submit_button = submit_col2.button("Submit", key="submit")

    if submit_button:
        dataset.has_submitted = dataset.check_if_field_empty(
            field, field_placeholder=place)

        if dataset.has_submitted:
            # TODO: Upload to database
            # st.success(""" Successfully created new dataset: {0}.
            #                 """.format(dataset.name))

            if dataset.save_dataset():

                success_place.success(
                    f"Successfully created **{dataset.name}** dataset")

                if dataset.insert_dataset():

                    success_place.success(
                        f"Successfully stored **{dataset.name}** dataset information in database")

                    # reset NewDataset class object
                    dataset = NewDataset(
                        get_random_string(length=8))

                else:
                    st.error(
                        f"Failed to stored **{dataset.name}** dataset information in database")
            else:
                st.error(
                    f"Failed to created **{dataset.name}** dataset")

    st.write(vars(dataset))
    # for img in session_state.new_dataset.dataset:
    #     st.image(img)


def main():

    # ****************** TESTING ***********************
    existing_dataset, dataset_table_column_names = query_dataset_list()
    # st.write(existing_dataset)
    dataset_name_list, dataset_dict = get_dataset_name_list(existing_dataset)

    show(dataset_dict["My Second Dataset"])
    # ****************** TESTING ***********************


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
