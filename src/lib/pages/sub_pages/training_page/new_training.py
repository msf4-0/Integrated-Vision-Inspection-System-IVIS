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
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass


from core.utils.code_generator import get_random_string
from core.utils.log import log_error, log_info  # logger
from data_manager.database_manager import init_connection
from path_desc import chdir_root
from project.project_management import Project
from training.training_management import NewTraining, NewTrainingPagination
from user.user_management import User
from pages.sub_pages.training_page.new_training_subpages import (new_training_infodataset,
                                                                 new_training_training_config,
                                                                 new_training_augmentation_config)
from pages.sub_pages.models_page import models_page
# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>

# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


# >>>> Variable Declaration >>>>


def index():
    chdir_root()  # change to root directory
    RELEASE = False

    # ****************** TEST ******************************
    if not RELEASE:
        log_info("At Training INDEX")

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

        # session_state.append_project_flag = ProjectPermission.ViewOnly

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
    st.write("## __Add New Training__")

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
        NewTrainingPagination.Model: models_page.existing_models,

        NewTrainingPagination.TrainingConfig: new_training_training_config.training_configuration,
        NewTrainingPagination.AugmentationConfig: new_training_augmentation_config.augmentation_configuration
    }
    session_state.new_training_progress_bar.progress(
        (session_state.new_training_pagination + 1) / 4)
    new_training_page[session_state.new_training_pagination]()

# # ************************* NEW TRAINING SECTION SUBMISSION HANDLERS **********************
#     if 'new_training_submission_handlers' not in session_state:
#         session_state.new_training_submission_handlers = {
#             NewTrainingPagination.InfoDataset: {
#                 'insert': session_state.new_training.insert_training_info,
#                 'update': None,
#                 'context': {
#                     'new_training_name': session_state.new_training_name
#                 },
#                 'name_key': 'new_training_name'
#             },
#             NewTrainingPagination.Model: {
#                 'insert': None,
#                 'update': None,
#                 'context': {
#                     # 'new_model_name': session_state.new_model_name TODO
#                 },
#                 'name_key': 'new_model_name'
#             },
#             NewTrainingPagination.TrainingConfig: {
#                 'insert': None,
#                 'update': None,
#                 'context': {
#                     'training_param': {}
#                 },
#                 'name_key': None
#             },
#             NewTrainingPagination.AugmentationConfig: {
#                 'insert': None,
#                 'update': None,
#                 'context': {
#                     # 'augmentation_param': session_state.new_training_name TODO
#                 },
#                 'name_key': 'new_training_name'
#             },
#         }

# ************************* NEW TRAINING SECTION PAGINATION BUTTONS **********************
    # Placeholder for Back and Next button for page navigation
    # new_training_section_back_button_place, _,\
    #     new_training_section_next_button_place = st.columns([1, 3, 1])

    # def insert_handler() -> bool:
    #     # return Boolean to proceed page change
    #     # run insert script for current page
    #     field_placeholder = session_state.new_training_place
    #     context = session_state.new_training_submission_handlers[session_state.new_training_pagination].get(
    #         'context')
    #     name_key = session_state.new_training_submission_handlers[session_state.new_training_pagination].get(
    #         'name_key')
    #     insert_flag = session_state.new_training.check_if_field_empty()

    #     # >>>> BACK BUTTON >>>>
    # if session_state.new_training_pagination > NewTrainingPagination.InfoDataset:

    #     def to_new_training_back_page():
    #         if session_state.new_training_pagination > NewTrainingPagination.InfoDataset:

    #             # Run submission according to current page
    #             # BACK page if constraints are met
    #             if session_state.new_training_submission_handlers[session_state.new_training_pagination]():
    #                 session_state.new_training_pagination -= 1

    #     with new_training_section_back_button_place:
    #         st.button("back", key="new_training_back_button",
    #                   on_click=to_new_training_back_page)

    # # >>>> NEXT BUTTON >>>>
    # if session_state.new_training_pagination < NewTrainingPagination.AugmentationConfig:
    #     def to_new_training_next_page():
    #         if session_state.new_training_pagination < NewTrainingPagination.AugmentationConfig:

    #             # Run submission according to current page
    #             # NEXT page if constraints are met
    #             if new_training_submission_handlers[session_state.new_training_pagination]():
    #                 session_state.new_training_pagination += 1

    #     with new_training_section_next_button_place:
    #         st.button("next", key="new_training_next_button",
    #                   on_click=to_new_training_next_page)
    log_info(
        f" New Training Pagination: {NewTrainingPagination(session_state.new_training_pagination)}")
    st.write(vars(session_state.new_training))


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        index()

    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
