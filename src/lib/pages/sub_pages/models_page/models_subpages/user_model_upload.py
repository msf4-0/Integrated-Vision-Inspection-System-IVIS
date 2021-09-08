"""
Title: New Models Page
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

from copy import deepcopy
import sys
from enum import IntEnum
from pathlib import Path
from time import sleep
from pandas.core import frame

import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state as session_state

# DEFINE Web APP page configuration
layout = 'wide'
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[5]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

from core.utils.code_generator import get_random_string
from core.utils.form_manager import remove_newline_trailing_whitespace
from core.utils.log import log_error, log_info  # logger
from core.utils.file_handler import list_files_in_archived
from data_manager.database_manager import init_connection
from path_desc import chdir_root, USER_DEEP_LEARNING_MODEL_UPLOAD_DIR
from training.model_management import NewModel, Model, Framework
from deployment.deployment_management import Deployment, DeploymentType
# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


# >>>> Variable Declaration >>>>
model_upload = {}  # store
place = {}
DEPLOYMENT_TYPE = ("", "Image Classification", "Object Detection with Bounding Boxes",
                   "Semantic Segmentation with Polygons", "Semantic Segmentation with Masks")


chdir_root()  # change to root directory


def user_model_upload_page():

    # >>>> INIT >>>>

    # ******** SESSION STATE ***********************************************************

    if "model_upload" not in session_state:
        session_state.model_upload = NewModel(get_random_string(length=8))

    # ******** SESSION STATE *********************************************************

    # Page title
    st.write("# __Upload Deep Learning Model__")
    st.markdown("___")

    # ************COLUMN PLACEHOLDERS *****************************************************
    # right-align the Model ID relative to the page
    id_blank, id_right = st.columns([3, 1])

    # Columns for Project Information
    infocol1, infocol2, infocol3 = st.columns([1.5, 3.5, 0.5])

    # Columns for Deployment Type
    DTcol1, DTcol2, DTcol3 = st.columns([1.5, 3.5, 0.5])

    # Columns for Framework
    framework_col1, framework_col2, framework_col3 = st.columns([
                                                                1.5, 3.5, 0.5])

    # Columns for Model Upload
    model_upload_col1, model_upload_col2, model_upload_col3 = st.columns([
                                                                         1.5, 3.5, 0.5])
    # Columns for Model Input Size
    model_input_size_col1, model_input_size_col2, model_input_size_col3 = st.columns([
        1.5, 3.5, 0.5])
    # ************COLUMN PLACEHOLDERS *****************************************************
    # <<<< INIT <<<<

# >>>> New Project INFO >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    id_right.write(
        f"### __Model ID:__ {session_state.model_upload.id}")

    infocol1.write("## __Model Information :__")

    # >>>> CHECK IF NAME EXISTS CALLBACK >>>>
    def check_if_name_exist(field_placeholder, conn):
        context = {'column_name': 'name',
                   'value': session_state.model_upload_name}

        if session_state.model_upload_name:
            if session_state.model_upload.check_if_exists(context, conn):
                session_state.model_upload.name = None
                field_placeholder['model_upload_name'].error(
                    f"Model name used. Please enter a new name")
                sleep(1)
                field_placeholder['model_upload_name'].empty()
                log_error(f"Model name used. Please enter a new name")
            else:
                session_state.model_upload.name = session_state.model_upload_name
                log_info(f"Model name fresh and ready to rumble")
        else:
            pass

    with infocol2:

        # ******************************** MODEL TITLE ********************************
        st.text_input(
            "Model Name", key="model_upload_name",
            help="Enter the name of the project",
            on_change=check_if_name_exist, args=(place, conn,))
        place["model_upload_name"] = st.empty()

        # ************************* PROJECT DESCRIPTION (OPTIONAL) *************************
        description = st.text_area(
            "Description (Optional)", key="new_project_desc",
            help="Enter the description of the project")

        if description:
            session_state.model_upload.desc = remove_newline_trailing_whitespace(
                description)
        else:
            pass

    # ******************************** DEPLOYMENT TYPE  ********************************
    DTcol1.write("## __Deployment Type :__")
    deployment_type_constant = None
    with DTcol2:
        st.selectbox(label="Deployment Type",
                     options=DEPLOYMENT_TYPE,
                     index=0,
                     format_func=lambda x: 'Select an option' if x == '' else x,
                     key="model_upload_deployment_type",
                     help="Please select the deployment type for the model")

        place["model_upload_deployment_type"] = st.empty()
        session_state.model_upload.deployment_type = session_state.model_upload_deployment_type

        if session_state.model_upload_deployment_type:
            deployment_type_constant = Deployment.get_deployment_type(
                deployment_type=session_state.model_upload.deployment_type,
                string=False)
    # ************************* FRAMEWORK *************************

    framework_col1.write("## __Framework :__")

    with framework_col2:
        framework_list_query = Model.get_framework_list()

        framework_list = [
            frameworks.Name for frameworks in framework_list_query]
        framework_list.insert(0, "")
        st.selectbox(label="Select Deep Learning Framework",
                     options=framework_list,
                     format_func=lambda x: 'Select a framework' if x == "" else x,
                     key='model_upload_framework')
        place['model_upload_framework'] = st.empty()

        session_state.model_upload.framework = session_state.model_upload_framework

    # ************************* MODEL UPLOAD *************************
    model_upload_col1.write("## __Model Upload :__")

    # ***********************ONLY ALLOW SINGLE FILE UPLOAD***********************
    with model_upload_col2:
        session_state.model_upload.file_upload = st.file_uploader(label='Upload Model',
                                                                  type=['zip', 'tar.gz',
                                                                        'tar.xz', 'tar.bz2'],
                                                                  accept_multiple_files=False,
                                                                  key='model_upload_widget')
        place['model_upload_file_upload'] = st.empty()
        # TODO AMMEND when adding compatibility for other Deep Learning Frameworks
        model_folder_structure_info = f"""
        ### Please ensure your files meets according to the following convention:
        #### 1. TensorFlow
        - Model Extension: `.pb` / `.h5`
        - Config Name: pipeline.config*
        - Labelmap Name: labelmap.pbtxt*

        *for Object Detection API (Object Detection with Bounding Boxes, Semantic Segmentation, Instance Segmentation)
        """
        with st.expander(label='Model Folder Structure'):
            st.info(model_folder_structure_info)

        if session_state.model_upload.file_upload:
            def check_files():
                with model_upload_col2:
                    session_state.model_upload.check_if_required_files_exist(
                        uploaded_file=session_state.model_upload.file_upload)

            st.button("Check compatibility",
                      key='check_files', on_click=check_files)
    # ************************* MODEL INPUT SIZE *************************
    # NOTE TO BE UPDATED FOR FUTURE UPDATES: VARIES FOR DIFFERENT DEPLOYMENT
    # IMAGE CLASSIFICATION, OBJECT DETECTION, IMAGE SEGMENTATION HAS SPECIFIC INPUT IMAGE SIZE
    input_size_context = {}

    if session_state.model_upload_deployment_type:

        if deployment_type_constant in [DeploymentType.Image_Classification, DeploymentType.OD,
                                        DeploymentType.Instance, DeploymentType.Semantic]:

            # Columns for Model Input Size
            _, model_input_size_title, _ = st.columns([1.5, 3.5, 0.5])
            _, model_input_size_col1, model_input_size_col2, model_input_size_col3, _ = st.columns([
                1.55, 1.2, 1.2, 1.2, 0.5])
            model_input_size_title.write(f"### Model Input Size")

            # with model_input_size_col1:
            #     st.number_input(
            #         label="Width (W)", key="model_input_width-", min_value=0, step=1)
            #     st.number_input(
            #         label="Height (H)", key="model_input_height-", min_value=0, step=1)
            #     st.number_input(
            #         label="Channels (C)", key="model_input_channel-", min_value=0, step=1)
            with model_input_size_col1:
                session_state.model_upload.model_input_size['width'] = st.number_input(
                    label="Width (W)", key="model_input_width", min_value=0, step=1)
                place['model_upload_width'] = st.empty()

            with model_input_size_col2:
                session_state.model_upload.model_input_size['height'] = st.number_input(
                    label="Height (H)", key="model_input_height", min_value=0, step=1)
                place['model_upload_height'] = st.empty()

            with model_input_size_col3:
                # NOTE OPTIONAL
                session_state.model_upload.model_input_size['channel'] = st.number_input(
                    label="Channels (C)", key="model_input_channel", min_value=0, step=1)
                place['model_upload_channel'] = st.empty()

            input_size_context = {
                'model_upload_width': session_state.model_upload.model_input_size['width'],
                'model_upload_height': session_state.model_upload.model_input_size['height'],
                'model_upload_channel': session_state.model_upload.model_input_size['channel'],
            }

        # NOTE KIV FOR OTHER DEPLOYMENTS
        # else:
        #     # Columns for Model Input Size
        #     # For Other deployments
        #     model_input_size_col1, model_input_size_col2, model_input_size_col3 = st.columns([
        #         1.5, 3.5, 0.5])
        place['model_upload_input_size'] = st.empty()
    # <<<<<<<< New Project INFO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ******************************** SUBMISSION *************************************************
    success_place = st.empty()
    context = {
        'model_upload_name': session_state.model_upload.name,
        'model_upload_deployment_type': session_state.model_upload.deployment_type,
        'model_upload_framework': session_state.model_upload.framework,
        'model_upload_file_upload': session_state.model_upload.file_upload,

    }

    submit_col1, submit_col2 = st.columns([3, 0.5])

    def model_upload_submit():

        # >>>> IF IT IS A NEW SUBMISSION
        if not session_state.model_upload.has_submitted:
            if session_state.model_upload.check_if_field_empty(context,
                                                               field_placeholder=place,
                                                               name_key='model_upload_name',
                                                               deployment_type_constant=deployment_type_constant,
                                                               input_size_context=input_size_context):
                pass

    submit_button_name = 'Submit' if session_state.model_upload.has_submitted == False else 'Update'
    # # TODO #72 Change to 'Update' when 'has_submitted' == True
    submit_button = submit_col2.button(
        label=submit_button_name, key="submit", on_click=model_upload_submit)

    # # >>>> Removed
    # # session_state.model_upload.has_submitted = False

    # col1, col2 = st.columns(2)
    # col1.write(vars(session_state.model_upload))
    # # col2.write(vars(session_state.new_editor))
    # col2.write(context)
    # # col2.write(dataset_dict)
    st.write(vars(session_state.model_upload))


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        user_model_upload_page()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
