"""
Title: Training Dashboard
Date: 30/8/2021
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
from enum import IntEnum
from pathlib import Path
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

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

from core.utils.log import log_error, log_info  # logger
from data_manager.database_manager import init_connection
from data_manager.data_table_component.data_table import data_table
from pages.sub_pages.training_page import new_training
from path_desc import chdir_root
from project.project_management import Project, ProjectPermission
from training.training_management import Training, TrainingPagination
# >>>> TEMP
from user.user_management import User

# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


# >>>> Variable Declaration >>>>

place = {}

PROGRESS_COLUMN_HEADER = {
    "Image Classification": 'Steps',
    "Object Detection with Bounding Boxes": 'Checkpoint / Steps',
    "Semantic Segmentation with Polygons": 'Checkpoint / Steps',
    "Semantic Segmentation with Masks": 'Checkpoint / Steps'
}

chdir_root()  # change to root directory


def dashboard():
    log_info(f"Top of Training Dashboard")
    st.write(f"### Dashboard")

    # >>>> QUERY PROJECT TRAINING >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # namedtuple of query from DB
    all_project_training, project_training_column_names = \
        Training.query_all_project_training(session_state.project.id,
                                            deployment_type=session_state.project.deployment_type,
                                            return_dict=True,
                                            for_data_table=True,
                                            progress_preprocessing=True)

    # ************************* SESSION STATE ************************************************
    if "training_dashboard_table" not in session_state:
        session_state.training_dashboard_table = None

    # ***************************COLUMN PLACEHOLDERS *****************************************
    create_new_training_button_col1 = st.empty()

    # ****************************** CREATE NEW PROJECT BUTTON ****************************************

    def to_new_training_page():

        session_state.training_pagination = TrainingPagination.New

        if "training_dashboard_table" in session_state:
            del session_state.training_dashboard_table

    create_new_training_button_col1.button(
        "Create New Training Session", key='create_new_training_from_training_dashboard',
        on_click=to_new_training_page, help="Create a new training session")



    # **************** DATA TABLE COLUMN CONFIG *********************************************************

    project_training_columns = [
        {
            'field': "id",
            'headerName': "ID",
            'headerAlign': "center",
            'align': "center",
            'flex': 50,
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
            'field': "Model Name",
            'headerAlign': "center",
            'align': "center",
            'flex': 120,
            'hideSortIcons': True,
        },
        {
            'field': "Base Model Name",
            'headerAlign': "center",
            'align': "center",
            'flex': 170,
            'hideSortIcons': True,
        },

        {
            'field': "Is Started",
            'headerAlign': "center",
            'align': "center",
            'flex': 80,
            'hideSortIcons': True,
            'type': 'boolean',
        },
        {
            'field': "Progress",
            'headerName': f"{PROGRESS_COLUMN_HEADER[session_state.project.deployment_type]}",
            'headerAlign': "center",
            'align': "center",
            'flex': 120,
            'hideSortIcons': True,
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
    training_dashboard_data_table_place = st.empty()

    with training_dashboard_data_table_place:
        data_table(all_project_training, project_training_columns,
                   checkbox=False, key='training_dashboard_table')


def index():
    RELEASE = False
    log_info("At Training Dashboard INDEX")
    # ****************** TEST ******************************
    if not RELEASE:

        # ************************TO REMOVE************************
        with st.sidebar.container():
            st.image("resources/MSF-logo.gif", use_column_width=True)
            st.title("Integrated Vision Inspection System", anchor='title')
            st.header(
                "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
            st.markdown("""___""")

        # ************************TO REMOVE************************
        project_id_tmp = 43
        log_info(f"Entering Project {project_id_tmp}")

        session_state.append_project_flag = ProjectPermission.ViewOnly

        if "project" not in session_state:
            session_state.project = Project(project_id_tmp)
            log_info("Inside")
        if 'user' not in session_state:
            session_state.user = User(1)
        # ****************************** HEADER **********************************************
        st.write(f"# {session_state.project.name}")

        project_description = session_state.project.desc if session_state.project.desc is not None else " "
        st.write(f"{project_description}")

        st.markdown("""___""")
        # ****************************** HEADER **********************************************
    st.write(f"## **Training Section:**")
    # ************************ TRAINING PAGINATION *************************
    training_page = {
        TrainingPagination.Dashboard: dashboard,
        TrainingPagination.New: new_training.index,
        TrainingPagination.Existing: None,
        TrainingPagination.NewModel: None
    }

    if 'training_pagination' not in session_state:
        session_state.training_pagination = TrainingPagination.Dashboard

    # >>>> RETURN TO ENTRY PAGE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    training_dashboard_back_button_place = st.empty()

    if session_state.training_pagination != TrainingPagination.Dashboard:

        def to_training_dashboard_page():
            # TODO #133 Add New Training Reset
            session_state.training_pagination = TrainingPagination.Dashboard

        training_dashboard_back_button_place.button("Back to Training Dashboard",
                                                    key="back_to_training_dashboard_page",
                                                    on_click=to_training_dashboard_page)

    else:
        training_dashboard_back_button_place.empty()

    log_info(
        f"Entering Training Page:{session_state.training_pagination}")

    # TODO #132 Add reset to training session state
    # >>>> MAIN FUNCTION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    training_page[session_state.training_pagination]()


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        index()

    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
