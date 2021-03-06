""" 
Title: New Training Index Page
Date: 23/7/2021
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
======================================================================================== """

import sys
from pathlib import Path
from time import sleep

import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state as session_state

# DEFINE Web APP page configuration
layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass


from core.utils.code_generator import get_random_string
from core.utils.log import logger  # logger
from data_manager.database_manager import init_connection
from path_desc import chdir_root
from project.project_management import Project
from training.training_management import NewTraining, NewTrainingPagination
from user.user_management import User
from pages.sub_pages.training_page.new_training_subpages import (new_training_infodataset,
                                                                 new_training_training_config,
                                                                 new_training_augmentation_config)
from pages.sub_pages.models_page import models_page
from pages.sub_pages.training_page import run_training_page
# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>

# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


# >>>> Variable Declaration >>>>


def index():
    logger.debug("[NAVIGATOR] At new_training.py INDEX")
    chdir_root()  # change to root directory
    RELEASE = True

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

        # session_state.append_project_flag = ProjectPermission.ViewOnly

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
        # ****************************** HEADER **********************************************

    # >>>> INIT >>>>

    # ******** SESSION STATE ***********************************************************

    if "new_training" not in session_state:
        session_state.new_training = NewTraining(get_random_string(
            length=8), session_state.project)
        # set random project ID before getting actual from Database

    if "new_training_pagination" not in session_state:
        session_state.new_training_pagination = NewTrainingPagination.InfoDataset

    if "new_training_place" not in session_state:
        session_state.new_training_place = {}

    if "new_training_progress_bar" not in session_state:
        session_state.new_training_progress_bar = st.progress(0)
    # ******** SESSION STATE *********************************************************

    # >>>> Page title
    if session_state.new_training_pagination == NewTrainingPagination.InfoDataset:
        st.write("## __Add New Training__")
    else:
        st.write(f"## Training Session: {session_state.new_training.name}")

    # ************COLUMN PLACEHOLDERS *****************************************************

    #  Deployment Type  Placeholder
    dt_place, id_right = st.columns([3, 1])
    # right-align the training ID relative to the page

    # ************COLUMN PLACEHOLDERS *****************************************************

    with dt_place:
        st.write("### __Deployment Type:__",
                 f"{session_state.project.deployment_type}")

    id_right.write(
        f"### __Training ID:__ {session_state.new_training.id}")

    # <<<< INIT <<<<

    # ************************ NEW TRAINING PAGINATION *************************
    new_training_page = {
        NewTrainingPagination.InfoDataset: new_training_infodataset.infodataset,
        NewTrainingPagination.Model: models_page.index,

        NewTrainingPagination.TrainingConfig: new_training_training_config.training_configuration,
        NewTrainingPagination.AugmentationConfig: new_training_augmentation_config.augmentation_configuration,
        NewTrainingPagination.Training: run_training_page.index
    }
    session_state.new_training_progress_bar.progress(
        (session_state.new_training_pagination + 1) / len(new_training_page))
    logger.debug("New Training Pagination:"
                 f" {NewTrainingPagination(session_state.new_training_pagination)}")
    new_training_page[session_state.new_training_pagination]()

    # ! DEBUGGING PURPOSE, REMOVE LATER
    # st.write("vars(session_state.new_training) = ")
    # st.write(vars(session_state.new_training))
    # st.write("vars(session_state.new_training.attached_model)")
    # st.write(vars(session_state.new_training.attached_model))
    # st.write("session_state.new_training.attached_model.get_path()")
    # st.write(session_state.new_training.attached_model.get_path())
    # st.write("Exists:", session_state.new_training.attached_model.get_path().exists())


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        index()

    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
