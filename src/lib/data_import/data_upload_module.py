"""
Title: Data Uploader
Date: 9/8/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
from time import sleep, perf_counter
from typing import Union, Dict
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state as session_state

# DEFINE Web APP page configuration
# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

for path in sys.path:
    if str(LIB_PATH) not in sys.path:
        sys.path.insert(0, str(LIB_PATH))  # ./lib
    else:
        pass

from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
from core.webcam import webcam_webrtc
from core.utils.helper import check_filetype
from data_manager.database_manager import init_connection
from data_manager.dataset_management import Dataset
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])

# >>>> Variable Declaration >>>>
place = {}  # PLACEHOLDER

# TODO #44


def data_uploader(dataset: Dataset, field_placeholder: Dict, key: str = None):

    chdir_root()  # change to root directory

    # ******** SESSION STATE ********
    if "new_dataset" not in session_state:
        # set random dataset ID before getting actual from Database
        session_state.data_source = "File Upload 📂"
    # ******** SESSION STATE ********


# TODO
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

    # >>>>>>>> New Dataset Upload >>>>>>>>
    # else:
    # pass
    # if 'webcam_flag' not in session_state:
    #     session_state.webcam_flag = False
    #     session_state.file_upload_flag = False
    #     # session_state.img1=True

    st.write("## __Dataset Upload:__")
    data_source_options = ["Webcam 📷", "File Upload 📂"]
    # col1, col2 = st.columns(2)

    data_source = st.radio(
        "Data Source", options=data_source_options, key="data_source_radio")
    data_source = data_source_options.index(data_source)

    outercol1, outercol2, outercol3 = st.columns([1.5, 2, 2])
    dataset_size_string = f"- ### Number of datas: **{session_state.new_dataset.dataset_size}**"
    dataset_filesize_string = f"- ### Total size of data: **{(session_state.new_dataset.calc_total_filesize()):.2f} MB**"
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
            session_state.new_dataset.dataset = None
        dataset_size_string = f"- ### Number of datas: **{session_state.new_dataset.dataset_size}**"
        dataset_filesize_string = f"- ### Total size of data: **{(session_state.new_dataset.calc_total_filesize()):.2f} MB**"

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
    # **** Submit Button ****
# TODO #20
    success_place = st.empty()
    field = [session_state.new_dataset.name, session_state.new_dataset.dataset]
    st.write(field)
    submit_col1, submit_col2 = st.columns([3, 0.5])
    submit_button = submit_col2.button("Submit", key="submit")

    if submit_button:
        session_state.new_dataset.has_submitted = session_state.new_dataset.check_if_field_empty(
            field, field_placeholder=place)

        if session_state.new_dataset.has_submitted:
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
    show()


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
