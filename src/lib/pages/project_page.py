"""
Title: Project Page
Date: 5/7/2021
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
import gc
import os
import sys
from pathlib import Path

import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state
import tensorflow as tf

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>
# DEFINE Web APP page configuration
layout = 'wide'
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

if 'setup' not in session_state:
    # to disable warning messages from OpenCV, must do this before import cv2
    os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
    os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            # limit memory growth to avoid memory issues
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,",
                  len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    session_state.setup = True
# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

from annotation.annotation_management import (Annotations, LabellingPagination,
                                              reset_editor_page)
from core.utils.log import logger  # logger
from data_manager.data_table_component.data_table import data_table
from data_manager.database_manager import init_connection
from path_desc import chdir_root
from project.project_management import (NewProject, NewProjectPagination,
                                        Project, ProjectPagination,
                                        ProjectPermission, query_all_projects)
from user.user_management import User
from training.training_management import NewTraining, Training
from deployment.deployment_management import Deployment
from data_manager.dataset_management import NewDataset

from pages.sub_pages.dataset_page.new_dataset import new_dataset
from pages.sub_pages.project_page import existing_project, new_project

# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])
# PAGE_OPTIONS = {"Dataset", "Project", "Deployment"}

# # <<<< Variable Declaration <<<<
chdir_root()  # change to root directory

# # TODO: #40 REMOVE SIDEBAR AFTER INTEGRATING INTO APP.PY
# with st.sidebar.container():
#     st.image("resources/MSF-logo.gif", use_column_width=True)

#     st.title("Integrated Vision Inspection System", anchor='title')
#     st.header(
#         "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
#     st.markdown("""___""")

# st.radio("", options=PAGE_OPTIONS, key="all_pages")


def dashboard():
    logger.debug("Top of project dashboard")
    st.write(f"# Project")
    st.markdown("___")

    # ********* QUERY PROJECT ********************************************
    # namedtuple of query from DB
    existing_project, project_table_column_names = query_all_projects(
        return_dict=True, for_data_table=True)
    # st.write(existing_project)
    # ********* QUERY DATASET ********************************************

    # ******** SESSION STATE *********************************************************

    if 'project_status' not in session_state:
        session_state.project_status = ProjectPagination.Existing
    else:
        session_state.project_status = ProjectPagination.Existing
    if 'append_project_flag' not in session_state:
        session_state.append_project_flag = ProjectPermission.ViewOnly

    if "all_project_table" not in session_state:
        session_state.all_project_table = None
    # ******** SESSION STATE *********************************************************

    # ************COLUMN PLACEHOLDERS *****************************************************
    create_new_project_button_col1 = st.empty()

    # ************COLUMN PLACEHOLDERS *****************************************************

    # ***************** CREATE NEW PROJECT BUTTON *********************************************************
    def to_new_project_page():

        session_state.project_pagination = ProjectPagination.New
        session_state.project_status = ProjectPagination.New

        if "all_project_table" in session_state:
            del session_state.all_project_table

    if session_state.project_pagination == ProjectPagination.Dashboard:
        create_new_project_button_col1.button(
            "Create New Project", key='create_new_project_from project_dashboard', on_click=to_new_project_page, help="Create new project")
    # ***************** CREATE NEW PROJECT BUTTON *********************************************************

    # **************** DATA TABLE COLUMN CONFIG *********************************************************
    project_columns = [
        {
            'field': "Name",
            'headerName': "Name",
            'headerAlign': "center",
            'align': "center",
            'flex': 130,
            'hideSortIcons': True,
        },
        {
            'field': "Description",
            'headerName': "Description",
            'headerAlign': "center",
            'align': "left",
            'flex': 150,
            'hideSortIcons': True,
        },
        {
            'field': "Deployment Type",
            'headerName': "Deployment Type",
            'headerAlign': "center",
            'align': "center",
            'flex': 150,
            'hideSortIcons': True,
        },
        {
            'field': "Date/Time",
            'headerName': "Date/Time",
            'headerAlign': "center",
            'align': "center",
            'flex': 100,
            'hideSortIcons': True,
            'type': 'date',
        },
    ]

    # **************** DATA TABLE COLUMN CONFIG *********************************************************

    # >>>>>>>>>>>> DATA TABLE CALLBACK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    data_table_place = st.empty()

    def to_existing_project():
        # clear out the annoying "Create New Project" button that never vanished from the shadow
        create_new_project_button_col1.empty()

        project_id_tmp = session_state.all_project_table[0]
        logger.info(f"Entering Project {project_id_tmp}")

        session_state.project_pagination = ProjectPagination.Existing
        session_state.project_status = ProjectPagination.Existing
        session_state.append_project_flag = ProjectPermission.ViewOnly

        if "project" not in session_state:
            session_state.project = Project(project_id_tmp)

        else:
            session_state.project = Project(project_id_tmp)

        data_table_place.empty()
        if "all_project_table" in session_state:
            del session_state.all_project_table
        st.experimental_rerun()

    if "all_project_table" not in session_state:
        session_state.all_project_table = None

    with data_table_place:
        data_table(existing_project, project_columns,
                   checkbox=False, key='all_project_table', on_change=to_existing_project)
        # st.write(session_state.all_project_table)


def index():

    project_page = {
        ProjectPagination.Dashboard: dashboard,
        ProjectPagination.New: new_project.index,
        ProjectPagination.Existing: existing_project.index,
        ProjectPagination.NewDataset: new_dataset
    }

    if 'project_pagination' not in session_state:
        session_state.project_pagination = ProjectPagination.Dashboard

    if 'project_status' not in session_state:
        session_state.project_status = None

    if 'user' not in session_state:
        session_state.user = User(1)

    # NOTE DEPRECATED ******************************************************************************
    # project_page_options = ("Dashboard", "Create New Project")
    # def project_page_navigator():

    #     NewProject.reset_new_project_page()

    #     session_state.project_pagination = project_page_options.index(
    #         session_state.project_page_navigator_radio)
    #     session_state.new_project_pagination = NewProjectPagination.Entry
    #     if "project_page_navigator_radio" in session_state:
    #         del session_state.project_page_navigator_radio

    # with navigator.expander("Project", expanded=True):
    #     st.radio("", options=project_page_options,
    #              index=session_state.project_pagination, on_change=project_page_navigator, key="project_page_navigator_radio")

    def to_project_dashboard():
        tf.keras.backend.clear_session()
        gc.collect()

        # reset all pages
        NewProject.reset_new_project_page()
        reset_editor_page()
        NewDataset.reset_new_dataset_page()
        NewTraining.reset_new_training_page()
        Training.reset_training_page()
        Project.reset_project_page()
        Project.reset_settings_page()
        Deployment.reset_deployment_page()

        session_state.project_pagination = ProjectPagination.Dashboard
        session_state.new_project_pagination = NewProjectPagination.Entry
        session_state.labelling_pagination = LabellingPagination.AllTask
        # if "project_page_navigator_radio" in session_state:
        #     del session_state.project_page_navigator_radio

    navigator = st.sidebar.empty()
    with navigator.container():
        st.button("Home", key="to_project_dashboard_sidebar",
                  on_click=to_project_dashboard,
                  help="This clears the current project session.")
    logger.debug(f"Navigator: {session_state.project_pagination = }")
    # st.write(session_state.project_pagination)
    project_page[session_state.project_pagination]()

    # st.write(session_state)


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        index()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
