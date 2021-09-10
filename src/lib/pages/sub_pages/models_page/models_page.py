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
from time import sleep
from copy import deepcopy
import pandas as pd
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state
from core.utils.form_manager import remove_newline_trailing_whitespace

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
from data_manager.data_table_component.data_table import data_table
from data_manager.database_manager import init_connection
from pages.sub_pages.models_page.models_subpages.user_model_upload import \
    user_model_upload_page
from path_desc import chdir_root
from project.project_management import Project
from training.model_management import Model, ModelsPagination, NewModel
from training.training_management import (NewTrainingSubmissionHandlers,
                                          Training)
from user.user_management import User

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration <<<<
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])
# <<<< Variable Declaration <<<<


def existing_models():
    log_info(f"At Existing Model Page")
    existing_models, existing_models_column_names = deepcopy(Model.query_model_table(for_data_table=True,
                                                                                     return_dict=True,
                                                                                     deployment_type=session_state.new_training.deployment_type))

    # st.write(vars(session_state.training))

    # ************************* SESSION STATE ************************************************
    if "existing_models_table" not in session_state:
        session_state.existing_models_table = None

    if 'training_model' not in session_state:
        session_state.new_training.training_model = NewModel()
    # ************************* SESSION STATE ************************************************

    # ************************ COLUMN PLACEHOLDER ********************************************
    to_model_upload_page_button_place = st.empty()
    st.write(
        f"**Step 1: Select a Deep Learning model from the table to be used for training:** ")

    main_col1, main_col2 = st.columns([3, 1])
    # ************************ COLUMN PLACEHOLDER ********************************************

    def to_model_upload_page():
        session_state.models_pagination = ModelsPagination.ModelUpload

        if "existing_models_table" not in session_state:
            del session_state.existing_models_table

    to_model_upload_page_button_place.button(label="Upload Deep Learning Model",
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

    # Returns model_id -> store inside model class
    def instantiate_model():
        model_df_row = Model.filtered_models_dataframe(models=existing_models,
                                                       dataframe_col="id",
                                                       filter_value=session_state.existing_models_table[0],
                                                       column_names=existing_models_column_names)

        # Instantiate Attached Model in `new_training` object
        session_state.new_training.attached_model = Model(
            model_row=model_df_row[0])

    with main_col1.container():
        data_table(rows=existing_models,
                   columns=existing_models_columns,
                   checkbox=False,
                   key='existing_models_table', on_change=instantiate_model)

    with main_col2.container():

        if session_state.new_training.attached_model:

            y = session_state.new_training.attached_model.get_perf_metrics()

            df_metrics = Model.create_perf_metrics_table(y)
            model_information = f"""
            ### Model Information:
            #### Name: {session_state.new_training.attached_model.name}
            #### Framework: {session_state.new_training.attached_model.framework}
            #### Model Input Size: {session_state.new_training.attached_model.model_input_size}

            """
            st.info(model_information)
            st.write(f"#### Metrics:")
            st.table(df_metrics)
            session_state.new_training_place["attached_model_selection"] = st.empty(
            )

            def check_if_name_exist(field_placeholder, conn):
                context = {'column_name': 'name',
                           'value': session_state.training_model_name}

                if session_state.training_model_name:
                    if session_state.new_training.training_model.check_if_exists(context, conn):
                        session_state.new_training.training_model.name = None
                        field_placeholder['training_model_name'].error(
                            f"Model name used. Please enter a new name")
                        sleep(0.7)
                        field_placeholder['training_model_name'].empty()
                        log_error(
                            f"Training Model name used. Please enter a new name")
                    else:
                        session_state.new_training.training_model.name = session_state.training_model_name
                        log_info(
                            f"Training Model name fresh and ready to rumble")
                else:
                    pass

            # ******************************** MODEL TITLE ********************************
            st.text_input('Model Name', key="training_model_name",
                          help="Enter the name of the model to be exported after training",
                          on_change=check_if_name_exist,
                          args=(session_state.new_training_place, conn,))
            session_state.new_training_place['training_model_name'] = st.empty(
            )

            # ************************* MODEL DESCRIPTION (OPTIONAL) *************************
            description = st.text_area(
                "Description (Optional)", key="training_model_desc",
                help="Enter the description of the model")

            if description:
                session_state.new_training.training_model.desc = remove_newline_trailing_whitespace(
                    description)
            else:
                pass


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
        if 'new_training' not in session_state:
            session_state.new_training = Training(training_id_tmp,
                                                  project=session_state.project)
        # ****************************** HEADER **********************************************
        st.write(f"# {session_state.project.name}")

        project_description = session_state.project.desc if session_state.project.desc is not None else " "
        st.write(f"{project_description}")

        st.markdown("""___""")
        new_training_progress_bar = st.progress(0)
        new_training_progress_bar.progress(2 / 4)
        # ****************************** HEADER **********************************************
        st.write(f"## **Training Section:**")

    # ************************ MODEL PAGINATION *************************
    models_page = {
        ModelsPagination.ExistingModels: existing_models,
        ModelsPagination.ModelUpload: user_model_upload_page
    }

    # ********************** SESSION STATE ******************************
    if 'models_pagination' not in session_state:
        session_state.models_pagination = ModelsPagination.ExistingModels
    if "new_training_place" not in session_state:
        session_state.new_training_place = {}

    # >>>> RETURN TO ENTRY PAGE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    existing_models_back_button_place = st.empty()

    if session_state.models_pagination != ModelsPagination.ExistingModels:

        def to_existing_model_page():

            session_state.models_pagination = ModelsPagination.ExistingModels
            # Reset all widget attributes in User Model Upload page
            NewModel.reset_model_upload_page()

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
    # ************************* NEW TRAINING SECTION PAGINATION BUTTONS **********************
    # Placeholder for Back and Next button for page navigation
    new_training_section_back_button_place, _, \
        new_training_section_next_button_place = st.columns([0.5, 3, 0.2])

    # typing.NamedTuple type
    # TODO
    new_training_model_submission_dict = NewTrainingSubmissionHandlers(
        insert=session_state.new_training.training_model.create_new_project_model_pipeline,
        update=session_state.new_training.training_model,
        context={
            'training_model_name': session_state.new_training.training_model.name,
            'attached_model_selection': session_state.existing_models_table
        },
        name_key='training_model_name'
    )

    # >>>> NEXT BUTTON >>>>
    # TODO

    def to_training_parameter_page():

        # run submission according to current page
        # NEXT page if all constraints met

        # >>>> IF IT IS A NEW SUBMISSION
        if not session_state.new_training.has_submitted[session_state.new_training_pagination]:

            if session_state.new_training.training_model.check_if_field_empty(
                new_training_model_submission_dict.context,
                field_placeholder=session_state.new_training_place,
                    name_key=new_training_model_submission_dict.name_key):
                # run create new project model pipeline
                # run update training attached model
                # set has_submitted =True
                pass

    with new_training_section_next_button_place:
        st.button("next", key="new_training_next_button",
                  on_click=None)

    st.write(vars(session_state.new_training.training_model))


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        index()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
