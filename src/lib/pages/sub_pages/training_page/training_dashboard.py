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
import pandas as pd

import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state as session_state

from training.model_management import ModelsPagination


# DEFINE Web APP page configuration
# layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

from core.utils.log import logger  # logger
from data_manager.database_manager import init_connection
from data_manager.data_table_component.data_table import data_table
from pages.sub_pages.training_page import new_training
from pages.sub_pages.models_page import models_page
from path_desc import chdir_root
from project.project_management import Project, ProjectPermission
from training.training_management import NewTraining, NewTrainingPagination, Training, TrainingPagination
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
    "Semantic Segmentation with Polygons": 'Checkpoint / Steps'
}

chdir_root()  # change to root directory


def dashboard():
    logger.debug(f"Top of Training Dashboard")
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

    existing_annotations, _ = session_state.project.query_annotations()
    if len(existing_annotations) >= 10:
        def to_new_training_page():
            session_state.training_pagination = TrainingPagination.New
            NewTraining.reset_new_training_page()

        create_new_training_button_col1.button(
            "Create New Training Session", key='create_new_training_from_training_dashboard',
            on_click=to_new_training_page, help="Create a new training session")
    else:
        st.warning("""Not enough annotations found for this project yet. Please go to the
        **Labelling** page and label for at least 10 images first before entering here.
        But note that 10 is only the minimum number of data to be used for a test run 😆""")

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

    def to_training_page():
        training_id_tmp = session_state.training_dashboard_table[0]
        logger.debug(f"Entering Training {training_id_tmp}")

        # if data_table is clicked on, that means there is already Training details stored in db
        # so just create Training instance instead of NewTraining instance
        session_state.new_training = Training(
            training_id_tmp, project=session_state.project)
        logger.debug("Training instance created successfully")

        # this is True because the user must have already submitted the info page to
        # have already had the data in the database
        session_state.new_training.has_submitted[NewTrainingPagination.InfoDataset] = True
        # moving to next page
        session_state.training_pagination = TrainingPagination.Existing
        logger.debug(
            f"Setting `training_pagination` to {session_state.training_pagination}")

        # set this to directly move to the `models_page`, this `new_training_pagination` is defined
        # in the new_training.py script
        if not session_state.new_training.attached_model:
            session_state.new_training_pagination = NewTrainingPagination.Model

        elif session_state.new_training.attached_model \
                and not session_state.new_training.training_param_dict:
            # model information form has already been submitted and stored in DB before
            session_state.new_training.has_submitted[NewTrainingPagination.Model] = True
            # set to this move directly to training_config page
            # session_state.models_pagination = ModelsPagination.TrainingConfig
            session_state.new_training_pagination = NewTrainingPagination.TrainingConfig
        else:
            for k in session_state.new_training.has_submitted.keys():
                # all forms have already been submitted before
                session_state.new_training.has_submitted[k] = True
            # set to this move directly to training page
            session_state.new_training_pagination = NewTrainingPagination.Training

        logger.debug("Setting `new_training_pagination` to "
                     f"{session_state.new_training_pagination}")

        st.experimental_rerun()

    with training_dashboard_data_table_place:
        data_table(all_project_training, project_training_columns,
                   checkbox=False, key='training_dashboard_table', on_change=to_training_page)

    st.markdown('___')


def index():
    RELEASE = True
    logger.debug("Navigator: At training_dashboard.py INDEX")
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
        project_id_tmp = 4
        logger.debug(f"Entering Project {project_id_tmp}")

        session_state.append_project_flag = ProjectPermission.ViewOnly

        if "project" not in session_state:
            session_state.project = Project(project_id_tmp)
            logger.debug("Inside")
        if 'user' not in session_state:
            session_state.user = User(1)
        # ****************************** HEADER **********************************************
        st.write(f"# {session_state.project.name}")

        project_description = session_state.project.desc if session_state.project.desc is not None else " "
        st.write(f"{project_description}")

        st.markdown("""___""")
    # ****************** TEST ******************************

    # ****************************** HEADER **********************************************
    st.write(f"## **Training Section:**")
    # ************************ TRAINING PAGINATION *************************
    training_page = {
        TrainingPagination.Dashboard: dashboard,
        TrainingPagination.New: new_training.index,

        # although same with TrainingPagination.New, but the new_training script will directly
        # link to next page by setting the `NewTrainingPagination` in this script before moving
        TrainingPagination.Existing: new_training.index,
        TrainingPagination.NewModel: None
    }

    if 'training_pagination' not in session_state:
        session_state.training_pagination = TrainingPagination.Dashboard

    # >>>> RETURN TO ENTRY PAGE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    training_dashboard_back_button_place = st.sidebar.empty()

    if session_state.training_pagination != TrainingPagination.Dashboard:

        def to_training_dashboard_page():
            NewTraining.reset_new_training_page()
            Training.reset_training_page()
            session_state.training_pagination = TrainingPagination.Dashboard

        training_dashboard_back_button_place.button("Back to Training Dashboard",
                                                    key="back_to_training_dashboard_page",
                                                    on_click=to_training_dashboard_page)

    else:
        training_dashboard_back_button_place.empty()

    # >>>> RETURN TO TRAINING PAGE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    training_page_back_place = st.sidebar.empty()

    # show btn if all forms are submitted and currently not in training page
    if 'new_training' in session_state and \
        all(session_state.new_training.has_submitted.values()) and \
            session_state.new_training_pagination != NewTrainingPagination.Training:
        def to_training_page():
            session_state.training_pagination = TrainingPagination.Existing
            session_state.new_training_pagination = NewTrainingPagination.Training

        training_page_back_place.button("Back to Start Training Page",
                                        key="back_to_training_page",
                                        on_click=to_training_page)
    else:
        training_page_back_place.empty()

    logger.debug(
        f"Entering Training Page: {session_state.training_pagination = }")

    # TODO #132 Add reset to training session state
    # >>>> MAIN FUNCTION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    training_page[session_state.training_pagination]()


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        index()

    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
