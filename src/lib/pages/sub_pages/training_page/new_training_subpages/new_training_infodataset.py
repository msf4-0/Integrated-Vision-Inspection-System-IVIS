""" Copyright (C) 2021 Selangor Human Resource Development Centre

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
from time import sleep

import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state

# DEFINE Web APP page configuration
layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[5]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass
# >>>> User-defined Modules >>>>
from core.utils.form_manager import remove_newline_trailing_whitespace
from core.utils.helper import create_dataframe, get_df_row_highlight_color
from core.utils.log import log_error, log_info  # logger
from data_manager.database_manager import init_connection
from path_desc import chdir_root

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration <<<<
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])

# <<<< Variable Declaration <<<<

chdir_root()


def infodataset():

    # ************COLUMN PLACEHOLDERS *****************************************************

    infocol1, infocol2, infocol3 = st.columns([1.5, 3.5, 0.5])

    info_dataset_divider = st.empty()

    # create 2 columns for "New Data Button"
    datasetcol1, datasetcol2, datasetcol3, _ = st.columns(
        [1.5, 1.75, 1.75, 0.5])

    # COLUMNS for Dataset Dataframe buttons
    _, dataset_button_col1, _, dataset_button_col2, _, dataset_button_col3, _ = st.columns(
        [1.5, 0.15, 0.5, 0.45, 0.5, 0.15, 2.25])

    # ************COLUMN PLACEHOLDERS *****************************************************

    # >>>> New Training INFO >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    infocol1.write("## __Training Information :__")

    # >>>> CHECK IF NAME EXISTS CALLBACK >>>>
    def check_if_name_exist(field_placeholder, conn):

        context = {'column_name': 'name',
                   'value': session_state.new_training_name}

        log_info(f"New Training: {context}")

        if session_state.new_training_name:
            if session_state.new_training.check_if_exists(context, conn):

                session_state.new_training.name = None
                field_placeholder['new_training_name'].error(
                    f"Training name used. Please enter a new name")
                sleep(1)
                field_placeholder['new_training_name'].empty()
                log_error(f"Training name used. Please enter a new name")

            else:
                session_state.new_training.name = session_state.new_training_name
                log_info(f"Training name fresh and ready to rumble")

        else:
            pass
   
    with infocol2:

        # **** TRAINING TITLE ****
        st.text_input(
            "Training Title", key="new_training_name",
            help="Enter the name of the training",
            on_change=check_if_name_exist, args=(session_state.new_training_place, conn,))
        session_state.new_training_place["new_training_name"] = st.empty()

        # **** TRAINING DESCRIPTION (Optional) ****
        description = st.text_area(
            "Description (Optional)", key="new_training_desc",
            help="Enter the description of the training")

        if description:
            session_state.new_training.desc = remove_newline_trailing_whitespace(
                description)
        else:
            pass

    # <<<<<<<< New Training INFO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        infodataset()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
