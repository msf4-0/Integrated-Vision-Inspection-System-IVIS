"""
Title: Models Page (Index)
Date: 5/9/2021
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

import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state

# DEFINE Web APP page configuration
layout = 'wide'
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass


# >>>> User-defined Modules >>>>

from core.utils.log import log_error, log_info  # logger
from data_import.models_upload_module import model_uploader
from data_manager.database_manager import init_connection
from data_table import data_table
from path_desc import chdir_root
from project.project_management import Project
from training.model_management import Model, ModelsPagination
from training.training_management import Training
from user.user_management import User

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration <<<<
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])
# <<<< Variable Declaration <<<<


def existing_models():
    log_info(f"At Existing Model Page")
    existing_models, existing_models_column_names = Model.query_model_table(for_data_table=True,
                                                                            return_dict=True,
                                                                            deployment_type=session_state.training.deployment_type)
    # st.write(vars(session_state.training))

    # ************************* SESSION STATE ************************************************
    if "existing_models_table" not in session_state:
        session_state.existing_models_table = None
    # ************************* SESSION STATE ************************************************

    def to_model_upload_page():
        session_state.models_pagination = ModelsPagination.ModelUpload

        if "existing_models_table" not in session_state:
            del session_state.existing_models_table

    st.button(label="Upload Deep Learning Model",
              key="upload_new_model",
              on_click=to_model_upload_page)
    # **************** DATA TABLE COLUMN CONFIG *********************************************************
    existing_models_columns = [
        {
            'field': "id",
            'headerName': "ID",
            'headerAlign': "center",
            'align': "center",
            'flex': 50,
            'hideSortIcons': True,

        },


        {
            'field': "Name",
            'headerAlign': "center",
            'align': "center",
            'flex': 120,
            'hideSortIcons': True,
        },
        {
            'field': "Framework",
            'headerAlign': "center",
            'align': "center",
            'flex': 120,
            'hideSortIcons': True,
        },
        {
            'field': "Model Type",
            'headerAlign': "center",
            'align': "center",
            'flex': 120,
            'hideSortIcons': True,
        },


        {
            'field': "Training Name",
            'headerAlign': "center",
            'align': "center",
            'flex': 150,
            'hideSortIcons': False,
        },
        {
            'field': "Date/Time",
            'headerAlign': "center",
            'align': "center",
            'flex': 100,
            'hideSortIcons': True,
            'type': 'date',
        },


    ]
    # >>>>>>>>>>>>>>>>>>>>>>>>>>> DATA TABLE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    data_table(rows=existing_models,
               columns=existing_models_columns,
               checkbox=False,
               key='existing_models_table')


def index():

    chdir_root()  # change to root directory
    RELEASE = False

    # ****************** TEST ******************************
    if not RELEASE:
        log_info("At Models INDEX")

        # ************************TO REMOVE************************
        with st.sidebar.container():
            st.image("resources/MSF-logo.gif", use_column_width=True)
            st.title("Integrated Vision Inspection System", anchor='title')
            st.header(
                "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
            st.markdown("""___""")

        # ************************TO REMOVE************************
        project_id_tmp = 43
        training_id_tmp = 18
        log_info(f"Entering Project {project_id_tmp}")

        # session_state.append_project_flag = ProjectPermission.ViewOnly

        if "project" not in session_state:
            session_state.project = Project(project_id_tmp)
            log_info("Inside")
        if 'user' not in session_state:
            session_state.user = User(1)
        if 'training' not in session_state:
            session_state.training = Training(training_id_tmp,
                                              project=session_state.project)
        # ****************************** HEADER **********************************************
        st.write(f"# {session_state.project.name}")

        project_description = session_state.project.desc if session_state.project.desc is not None else " "
        st.write(f"{project_description}")

        st.markdown("""___""")
        # ****************************** HEADER **********************************************
        st.write(f"## **Training Section:**")

    # ************************ MODEL PAGINATION *************************
    models_page = {
        ModelsPagination.ExistingModels: existing_models,
        ModelsPagination.ModelUpload: model_uploader
    }

    # ********************** SESSION STATE ******************************
    if 'models_pagination' not in session_state:
        session_state.models_pagination = ModelsPagination.ExistingModels
    if 'models_place' not in session_state:
        session_state.models_place = {}
    # >>>> RETURN TO ENTRY PAGE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    existing_models_back_button_place = st.empty()

    if session_state.models_pagination != ModelsPagination.ExistingModels:

        def to_existing_model_page():

            session_state.models_pagination = ModelsPagination.ExistingModels

        existing_models_back_button_place.button("Back to Existing Models Dashboard",
                                                 key="back_to_existing_models_page",
                                                 on_click=to_existing_model_page)

    else:
        existing_models_back_button_place.empty()

    log_info(
        f"Entering Models Page:{session_state.models_pagination}")

    st.write(f"### Models Selection")
    # >>>> MAIN FUNCTION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    models_page[session_state.models_pagination]()


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        index()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
