"""
Title: Model Upload Module
Date: 6/9/2021
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
from pathlib import Path
from time import perf_counter, sleep
from typing import Dict, Union
from copy import deepcopy
import streamlit as st
from humanize import naturalsize
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


from core.utils.log import log_error, log_info  # logger
from data_manager.database_manager import init_connection
from path_desc import chdir_root
from training.model_management import NewModel

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])

# >>>> Variable Declaration >>>>
place = {}  # PLACEHOLDER


def model_uploader():

    chdir_root()  # change to root directory

    st.write("## __Deep Learning Model Upload:__")

    st.markdown(" ____ ")

    # ******************************* MODEL UPLOAD WIDGET *************************************
    st.file_uploader(
        label="Upload Deep Learning Model",
        type=['zip', 'tar.gz', 'tar.xz', 'tar.bz2'],
        accept_multiple_files=False,
        key='model_upload_module')

    place["model_upload"] = st.empty()
    # ******************************* MODEL UPLOAD WIDGET *************************************
    st.markdown(" ____ ")

    # >>>>>>>>>>>>>>>>>>>>>>>> SUBMISSION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def update_dataset_callback(field, place):
        context = {'upload': field}
        dataset.has_submitted = dataset.check_if_field_empty(
            context, field_placeholder=place)

        if dataset.has_submitted:
            # st.success(""" Successfully created new dataset: {0}.
            #                 """.format(dataset.name))

            # set FLAG to 0
            session_state.append_data_flag = dataset.update_pipeline(
                success_place)

            if 'upload_module' in session_state:
                del session_state.upload_module
            # return session_state.append_data_flag

    success_place = st.empty()
    field = dataset.dataset

    st.button(
        "Submit", key="submit", on_click=update_dataset_callback, args=(field, place,))


def main():
    RELEASE = False

    # ****************** TEST ******************************
    if not RELEASE:

        if 'append_data_flag' not in session_state:
            session_state.append_data_flag = 0

        st.write(session_state.append_data_flag)

        with st.expander("", expanded=False):
            model_uploader(session_state.dataset)

        submit = st.button("Test", key="testing")
        if submit:
            with session_state.place['test'].container():
                st.success("Can")
                st.success("Yea")

        def flag_1():
            session_state.append_data_flag = 1
            log_info("Flag 1")

        def flag_0():
            session_state.append_data_flag = 0
            log_info("Flag 0")
            # if 'upload_widget' in session_state:
            #     del session_state.upload_widget

        st.button("Flag 0", key='flag0', on_click=flag_0)
        st.button("Flag 1", key='flag1', on_click=flag_1)


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
