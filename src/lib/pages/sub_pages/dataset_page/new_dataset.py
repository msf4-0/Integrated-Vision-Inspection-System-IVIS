"""
Title: New Dataset Page
Date: 7/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)

Copyright (C) 2021 Selangor Human Resource Development Centre

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. 

Copyright (C) 2021 Selangor Human Resource Development Centre
SPDX-License-Identifier: Apache-2.0
========================================================================================

"""

import sys
from copy import deepcopy
from enum import IntEnum
from pathlib import Path
from time import perf_counter, sleep
from typing import Dict, Union
from humanize import naturalsize
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state as session_state

# DEFINE Web APP page configuration
# layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)


# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

for path in sys.path:
    if str(LIB_PATH) not in sys.path:
        sys.path.insert(0, str(LIB_PATH))  # ./lib
    else:
        pass

from core.utils.code_generator import get_random_string
from core.utils.helper import check_filetype
from core.utils.log import log_error, log_info  # logger
from core.webcam import webcam_webrtc
from data_manager.database_manager import init_connection
from data_manager.dataset_management import DatasetPagination, NewDataset
from path_desc import chdir_root
from project.project_management import NewProjectPagination, ProjectPagination

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


def new_dataset():

    chdir_root()  # change to root directory

    # ******** SESSION STATE ********

    if "new_dataset" not in session_state:
        # set random dataset ID before getting actual from Database
        log_info("Enter new dataset")
        session_state.new_dataset = NewDataset(get_random_string(length=8))
        session_state.data_source_radio = "File Upload 📂"
    # ******** SESSION STATE ********

    # >>>>>>>> New Dataset INFO >>>>>>>>
    # Page title
    st.write("# __Add New Dataset__")
    st.markdown("___")

    # right-align the dataset ID relative to the page
    _, id_right = st.columns([3, 1])
    id_right.write(
        f"### __Dataset ID:__ {session_state.new_dataset.dataset_id}")

    outercol1, outercol2, outercol3 = st.columns([1.5, 3.5, 0.5])

    # >>>>>>> DATASET INFORMATION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    outercol1.write("## __Dataset Information :__")

    # >>>> CHECK IF NAME EXISTS CALLBACK >>>>
    def check_if_name_exist(field_placeholder, conn):
        context = {'column_name': 'name', 'value': session_state.name}
        if session_state.name:
            if session_state.new_dataset.check_if_exists(context, conn):
                session_state.new_dataset.name = None
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

    # <<<<<<<< New Dataset INFO <<<<<<<<

    # >>>>>>>> New Dataset Upload >>>>>>>>

    outercol1, outercol2, outercol3 = st.columns([1.5, 3.5, 0.5])

    outercol1.write("## __Dataset Upload:__")
    data_source_options = ["Webcam 📷", "File Upload 📂"]
    # col1, col2 = st.columns(2)

    data_source = outercol2.radio(
        "Data Source", options=data_source_options, key="data_source_radio")
    data_source = data_source_options.index(data_source)

    outercol1, outercol2, outercol3 = st.columns([1.5, 2, 2])
    dataset_size_string = f"- ### Number of datas: **{session_state.new_dataset.dataset_size}**"
    dataset_filesize_string = f"- ### Total size of data: **{naturalsize(value=session_state.new_dataset.calc_total_filesize(),format='%.2f')}**"
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
    # TODO #24 Add other filetypes based on filetype table
    # Done #24

    elif data_source == 1:

        # uploaded_files_multi = outercol2.file_uploader(
        #     label="Upload Image", type=['jpg', "png", "jpeg", "mp4", "mpeg", "wav", "mp3", "m4a", "txt", "csv", "tsv"], accept_multiple_files=True, key="upload_widget", on_change=check_filetype_category, args=(place,))
        uploaded_files_multi = outercol2.file_uploader(
            label="Upload Image", type=['jpg', "png", "jpeg", "mp4", "mpeg", "wav", "mp3", "m4a", "txt", "csv", "tsv"], accept_multiple_files=True, key="upload_widget")
        # ******** INFO for FILE FORMAT **************************************
        with outercol1.expander("File Format Infomation", expanded=True):
            file_format_info = """
            1. Image: .jpg, .png, .jpeg
            2. Video: .mp4, .mpeg
            3. Audio: .wav, .mp3, .m4a
            4. Text: .txt, .csv
            """
            st.info(file_format_info)
        place["upload"] = outercol2.empty()
        if uploaded_files_multi:
            # outercol2.write(uploaded_files_multi) # TODO Remove
            check_filetype(
                uploaded_files_multi, session_state.new_dataset, place)

            session_state.new_dataset.dataset = deepcopy(uploaded_files_multi)

            session_state.new_dataset.dataset_size = len(
                uploaded_files_multi)  # length of uploaded files
        else:
            session_state.new_dataset.filetype = None
            session_state.new_dataset.dataset_size = 0  # length of uploaded files
            session_state.new_dataset.dataset = []
        dataset_size_string = f"- ### Number of datas: **{session_state.new_dataset.dataset_size}**"
        dataset_filesize_string = f"- ### Total size of data: **{naturalsize(value=session_state.new_dataset.calc_total_filesize(),format='%.2f')}**"

        # outercol2.write(uploaded_files_multi[0]) # TODO: Remove
        dataset_size_place.write(dataset_size_string)
        dataset_filesize_place.write(dataset_filesize_string)

    # Placeholder for WARNING messages of File Upload widget

    # with st.expander("Data Viewer", expanded=False):
    #     imgcol1, imgcol2, imgcol3 = st.columns(3)
    #     imgcol1.checkbox("img1", key="img1")
    #     for image in uploaded_files_multi:
    #         imgcol1.image(uploaded_files_multi[1])

    # TODO: KIV

    # col1, col2, col3 = st.columns([1, 1, 7])
    # webcam_button = col1.button(
    #     "Webcam 📷", key="webcam_button", on_click=update_webcam_flag)
    # file_upload_button = col2.button(
    #     "File Upload 📂", key="file_upload_button", on_click=update_file_uploader_flag)

    # <<<<<<<< New Dataset Upload <<<<<<<<
    # ******************************** SUBMISSION *************************************************
    success_place = st.empty()
    context = {'name': session_state.new_dataset.name,
               'upload': session_state.new_dataset.dataset}

    st.write(context)
    submit_col1, submit_col2 = st.columns([3, 0.5])
    submit_button = submit_col2.button("Submit", key="submit")

    if submit_button:
        keys = ["name", "upload"]
        session_state.new_dataset.has_submitted = session_state.new_dataset.check_if_field_empty(
            context, field_placeholder=place, name_key='name')

        if session_state.new_dataset.has_submitted:

            if session_state.new_dataset.save_dataset():

                success_place.success(
                    f"Successfully created **{session_state.new_dataset.name}** dataset")

                if session_state.new_dataset.insert_dataset():

                    success_place.success(
                        f"Successfully stored **{session_state.new_dataset.name}** dataset information in database")

                else:
                    st.error(
                        f"Failed to stored **{session_state.new_dataset.name}** dataset information in database")
            else:
                st.error(
                    f"Failed to created **{session_state.new_dataset.name}** dataset")


    st.write(vars(session_state.new_dataset))



def main():
    new_dataset()


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
