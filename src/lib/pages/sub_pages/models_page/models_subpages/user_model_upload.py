"""
Title: User Deep Learning Model Upload Page
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
from time import sleep
from typing import Any, Dict, List

import streamlit as st
import pandas as pd
from streamlit import cli as stcli
from streamlit import session_state

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>
# DEFINE Web APP page configuration
# layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# SRC = Path(__file__).resolve().parents[5]  # ROOT folder -> ./src
# LIB_PATH = SRC / "lib"
# if str(LIB_PATH) not in sys.path:
#     sys.path.insert(0, str(LIB_PATH))  # ./lib

from core.utils.code_generator import get_random_string
from core.utils.file_handler import (list_files_in_archived,
                                     save_uploaded_extract_files)
from core.utils.form_manager import remove_newline_trailing_whitespace
from core.utils.log import logger
from data_manager.database_manager import init_connection
from deployment.deployment_management import COMPUTER_VISION_LIST, Deployment, DeploymentType
from path_desc import USER_DEEP_LEARNING_MODEL_UPLOAD_DIR, chdir_root
from training.model_management import Model, ModelsPagination, NewModel, ModelCompatibility
from training.labelmap_management import Labels
from training.labelmap_generator import labelmap_generator
# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


# >>>> Variable Declaration >>>>
model_upload = {}  # store
place = {}
DEPLOYMENT_TYPE = ("", "Image Classification", "Object Detection with Bounding Boxes",
                   "Semantic Segmentation with Polygons")


chdir_root()  # change to root directory


def user_model_upload_page():
    # >>>> INIT >>>>

    # ******** SESSION STATE ***********************************************************

    if "model_upload" not in session_state:
        session_state.model_upload = NewModel(get_random_string(length=8))

    if 'labelmap' not in session_state:
        session_state.labelmap = Labels()

    if 'generate_labelmap_flag' not in session_state:
        session_state.generate_labelmap_flag = False

    model_upload: NewModel = session_state.model_upload
    labelmap: Labels = session_state.labelmap

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
    # Columns for Labelmap Dataframe
    labelmap_col1, labelmap_col2, labelmap_col3 = st.columns([
        1.5, 3.5, 0.5])
    # Columns for Model Input Size
    model_input_size_col1, model_input_size_col2, model_input_size_col3 = st.columns([
        1.5, 3.5, 0.5])
    # ************COLUMN PLACEHOLDERS *****************************************************
    # <<<< INIT <<<<

    # >>>> New Project INFO >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    id_right.write(
        f"### __Model ID:__ {model_upload.id}")

    infocol1.write("## __Model Information :__")

    # >>>> CHECK IF NAME EXISTS CALLBACK >>>>
    def check_if_name_exist(field_placeholder, conn):
        context = {'column_name': 'name',
                   'value': session_state.model_upload_name}

        if session_state.model_upload_name:
            if model_upload.check_if_exists(context, conn):
                model_upload.name = None
                field_placeholder['model_upload_name'].error(
                    f"Model name used. Please enter a new name")
                sleep(1)
                field_placeholder['model_upload_name'].empty()
                logger.error(f"Model name used. Please enter a new name")
            else:
                model_upload.name = session_state.model_upload_name
                logger.info(f"Model name fresh and ready to rumble")

    with infocol2:
        # ******************************** MODEL TITLE ********************************
        st.text_input(
            "Model Name", key="model_upload_name",
            help="Enter the name of the project",
            on_change=check_if_name_exist, args=(place, conn,))
        place["model_upload_name"] = st.empty()

        # ************************* MODEL DESCRIPTION (OPTIONAL) *************************
        description = st.text_area(
            "Description (Optional)", key="model_upload_desc",
            help="Enter the description of the project")

        if description:
            model_upload.desc = remove_newline_trailing_whitespace(
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
        model_upload.deployment_type = session_state.model_upload_deployment_type

        if session_state.model_upload_deployment_type:
            deployment_type_constant = Deployment.get_deployment_type(
                deployment_type=model_upload.deployment_type,
                string=False)
            logger.debug(f"{deployment_type_constant = }")
    # ************************* FRAMEWORK *************************

    # framework_col1.write("## __Framework :__")

    with framework_col2:
        # framework_list_query = Model.get_framework_list()

        # framework_list = [
        #     frameworks.Name for frameworks in framework_list_query]
        # framework_list.insert(0, "")
        # st.selectbox(label="Select Deep Learning Framework",
        #              options=framework_list,
        #              format_func=lambda x: 'Select a framework' if x == "" else x,
        #              key='model_upload_framework')
        # place['model_upload_framework'] = st.empty()

        # NOTE: using only TensorFlow framework for now
        # model_upload.framework = session_state.model_upload_framework
        model_upload.framework = 'TensorFlow'

    # ************************* MODEL UPLOAD *************************
    model_upload_col1.write("## __Model Upload :__")

    # ***********************ONLY ALLOW SINGLE FILE UPLOAD***********************
    def check_output_nodes(labelmap_dict: List[Dict[int, Any]], num_output_nodes: int):
        """Check whether number of nodes of output layer is the same as the number of
        classes in the labelmap file. Only check for uploaded classification/segmentation
        Keras Model. Executes `st.stop()` if they are not equal."""
        if model_upload.deployment_type != 'Object Detection with Bounding Boxes':
            labelmap_num_classes = len(labelmap_dict)
            if labelmap_num_classes != num_output_nodes:
                st.error(
                    """Number of classes found in labelmap file does not match the 
                    number of nodes in the output layer of the uploaded Keras Model.""")
                logger.error(
                    """Number of classes found in labelmap file does not match the 
                    number of nodes in the output layer of the uploaded Keras Model.""")
                st.stop()

    with model_upload_col2:
        uploaded_file = st.file_uploader(label='Upload Model',
                                         type=['zip', 'tar.gz',
                                               'tar.xz', 'tar.bz2'],
                                         accept_multiple_files=False,
                                         key='model_upload_widget')
        place['model_upload_file_upload'] = st.empty()
        # TODO AMMEND when adding compatibility for other Deep Learning Frameworks
        if deployment_type_constant == DeploymentType.OD:
            model_folder_structure_info = f"""
            ### Please ensure your files meet the following convention for TensorFlow Object Detection API:
            An archive file (zipfile or tarfile) containing:
            - Model file with extension: `.pb`
            - Config file: pipeline.config
            - Labelmap file: labelmap.pbtxt (optional)

            The directory structure within the archived file should be exactly as 
            follow to avoid complications:
            """
            tree = """
            <archive_root>
            ├── checkpoint
            │   ├── checkpoint
            │   ├── ckpt-0.data-00000-of-00001
            │   └── ckpt-0.index
            ├── labelmap.pbtxt
            ├── pipeline.config
            └── saved_model
                ├── assets
                ├── saved_model.pb
                └── variables
                    ├── variables.data-00000-of-00001
                    └── variables.index
            """
            with st.expander(label='Model Folder Structure'):
                st.markdown(model_folder_structure_info)
                st.code(tree, language='bash')
        elif deployment_type_constant in (DeploymentType.Image_Classification,
                                          DeploymentType.Instance):
            task_dict = {DeploymentType.Image_Classification: 'Image Classification',
                         DeploymentType.Instance: 'Semantic Segmentation'}
            task_name = task_dict[deployment_type_constant]
            model_folder_structure_info = f"""
            ### Please ensure your files in the uploaded archive meet the following convention:
            #### TensorFlow Keras H5 Model
            - A Keras model file with the following extension: `.h5`
            - The model should be built and trained for the task of **{task_name}**
            #### Labelmap file (optional)
            - The extension should end with '.pbtxt'
            - If this file is provided, the model can be instantly deployed.
            """
            with st.expander(label='Model Folder Structure'):
                st.markdown(model_folder_structure_info)

        # **************************** CHECK UPLOADED MODELS COMPATIBILITY ****************************
        def check_files():
            context = {
                'model_upload_deployment_type': model_upload.deployment_type,
                # 'model_upload_framework': model_upload.framework,
                'model_upload_file_upload': uploaded_file
            }
            if model_upload.check_if_field_empty(context,
                                                 field_placeholder=place,
                                                 name_key='model_upload_name',
                                                 deployment_type_constant=deployment_type_constant):
                with st.spinner("Checking compatible files in uploaded model ..."):
                    num_output_nodes, label_map_files = model_upload.check_if_required_files_exist(
                        uploaded_file=uploaded_file)

                if model_upload.compatibility_flag <= 1:
                    model_upload.file_upload = uploaded_file

                if label_map_files:
                    labelmap.filename = label_map_files[0]
                    logger.debug(f"{labelmap.filename = }")
                    with st.spinner(text='Loading Labelmap'):
                        if model_upload.framework and model_upload.deployment_type:
                            label_map_string = labelmap.get_labelmap_member_from_archived(
                                name=labelmap.filename,
                                archived_filepath=model_upload.file_upload.name,
                                file_object=model_upload.file_upload)
                            # logger.info(label_map_string)
                            if label_map_string:
                                try:
                                    labelmap.dict = labelmap.generate_labelmap_dict(
                                        label_map_string=label_map_string,
                                        framework=model_upload.framework)
                                except Exception as e:
                                    logger.error(
                                        f"Error reading the uploaded labelmap file: {e}")
                                    st.error(
                                        "Error reading the uploaded labelmap file")
                                # logger.info(
                                #     f"labelmap_dict:{labelmap.dict}")
                                check_output_nodes(
                                    labelmap.dict, num_output_nodes)
                else:
                    # CLEAR LABELMAP DICT IF LABELMAP FILES DOES NOT EXISTS
                    labelmap.dict = {}

        if st.button("Check compatibility", key='check_files'):  # NOTE KIV
            check_files()

        # st.write("labelmap.dict")
        # st.write(labelmap.dict)
        # st.write(f"{len(labelmap.dict) = }")

        # *********************************TEMP*********************************

        # def save_file():
        #     if uploaded_file:
        #         with model_upload_col2:
        #             with st.spinner(text='Storing uploaded model'):
        #                 save_uploaded_extract_files(dst='/home/rchuzh/Desktop/test2',
        #                                             filename=uploaded_file.name,
        #                                             fileObj=uploaded_file)

        # st.button("Save file", key='save_file', on_click=save_file)

        # *********************************TEMP*********************************

    # *********************************************** SHOW TABLE OF LABELS ***********************************************
    if not labelmap.dict and model_upload.compatibility_flag not in (
            ModelCompatibility.Compatible, ModelCompatibility.MissingModel):
        with labelmap_col2.container():
            (session_state.generate_labelmap_flag,
             labelmap.label_map_string) = labelmap_generator(
                framework=model_upload.framework,
                deployment_type=model_upload.deployment_type)

            # TODO create labelmap file and move to dst folder

    if labelmap.dict and uploaded_file:
        if uploaded_file.name == model_upload.file_upload.name:
            with labelmap_col2.container():
                df = pd.DataFrame(labelmap.dict)
                df.set_index('id')
                st.write(f"Labelmap from Model:")
                st.dataframe(df)

    # *********************************************** SHOW TABLE OF LABELS ***********************************************

    # ************************* MODEL INPUT SIZE *************************
    # NOTE: not using model_input_size for now as I believe it's not necessary
    # REASON: TFOD training depends on the pipeline.config file for image_resizer.
    # Keras H5 models input shape can be changed later without affecting trained weights
    #  due to the nature of CNN that can accept any kind of input sizes.

    # NOTE TO BE UPDATED FOR FUTURE UPDATES: VARIES FOR DIFFERENT DEPLOYMENT
    # IMAGE CLASSIFICATION, OBJECT DETECTION, IMAGE SEGMENTATION HAS SPECIFIC INPUT IMAGE SIZE
    # input_size_context = {}
    # _, model_input_size_title, _ = st.columns([1.5, 3.5, 0.5])
    # _, model_input_size_col1, model_input_size_col2, model_input_size_col3, _ = st.columns([
    #     1.55, 1.2, 1.2, 1.2, 0.5])
    # if session_state.model_upload_deployment_type:

    #     if deployment_type_constant in COMPUTER_VISION_LIST:

    #         # Columns for Model Input Size
    #         model_input_size_title.write(f"### Model Input Size")

    #         # *******************************************************************************************
    #         # NOTE KIV Design decision
    #         # with model_input_size_col1:
    #         #     st.number_input(
    #         #         label="Width (W)", key="model_input_width-", min_value=0, step=1)
    #         #     st.number_input(
    #         #         label="Height (H)", key="model_input_height-", min_value=0, step=1)
    #         #     st.number_input(
    #         #         label="Channels (C)", key="model_input_channel-", min_value=0, step=1)
    #         with st.container():
    #             with model_input_size_col1:
    #                 model_upload.model_input_size['width'] = st.number_input(
    #                     label="Width (W)", key="model_input_width", min_value=0, step=1)
    #                 place['model_upload_width'] = st.empty()

    #             with model_input_size_col2:
    #                 model_upload.model_input_size['height'] = st.number_input(
    #                     label="Height (H)", key="model_input_height", min_value=0, step=1)
    #                 place['model_upload_height'] = st.empty()

    #             with model_input_size_col3:
    #                 # NOTE OPTIONAL
    #                 model_upload.model_input_size['channel'] = st.number_input(
    #                     label="Channels (C)", key="model_input_channel", min_value=0, step=1)
    #                 place['model_upload_channel'] = st.empty()

    #             input_size_context = {
    #                 'model_upload_width': model_upload.model_input_size['width'],
    #                 'model_upload_height': model_upload.model_input_size['height'],
    #                 'model_upload_channel': model_upload.model_input_size['channel'],
    #             }
    #     else:
    #         input_size_context = {}
        # *******************************************************************************************
        # NOTE KIV FOR OTHER DEPLOYMENTS
        # else:
        #     # Columns for Model Input Size
        #     # For Other deployments
        #     model_input_size_col1, model_input_size_col2, model_input_size_col3 = st.columns([
        #         1.5, 3.5, 0.5])
        # *******************************************************************************************

        place['model_upload_input_size'] = st.empty()
    # <<<<<<<< New Project INFO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ******************************** SUBMISSION *************************************************
    context = {
        'model_upload_name': model_upload.name,
        'model_upload_deployment_type': model_upload.deployment_type,
        # 'model_upload_framework': model_upload.framework,
        'model_upload_file_upload': uploaded_file,
    }

    # Columns for submit button
    ignore_col1, ignore_col2 = st.columns([2, 0.5])
    submit_col1, submit_col2 = st.columns([2, 0.5])
    with submit_col2:
        submit_btn_place = st.empty()
    bottom_col1, bottom_col2, bottom_col3 = st.columns([
        1.5, 3.5, 0.5])

    if not model_upload.has_submitted:
        def model_upload_submit():
            success = False
            # >>>> IF IT IS A NEW SUBMISSION
            if model_upload.check_if_field_empty(
                    context,
                    field_placeholder=place,
                    name_key='model_upload_name',
                    deployment_type_constant=deployment_type_constant):
                # input_size_context=input_size_context):

                logger.debug(
                    f"{model_upload.compatibility_flag = }")

                with model_upload_col2:
                    # NOTE: only image classification and segmentation have num_output_nodes
                    num_output_nodes, label_map_files = model_upload.check_if_required_files_exist(
                        uploaded_file=uploaded_file)
                    sleep(0.5)

                if model_upload.compatibility_flag == ModelCompatibility.MissingExtraFiles_ModelExists \
                        and not label_map_files:
                    # IF THERE ARE NON COMPULSORY FILES MISSING
                    model_upload.file_upload = uploaded_file

                    def continue_upload():
                        # CALLBACK to continue upload model to server disregarding the warning
                        model_upload.create_new_model_pipeline()

                    if not session_state.generate_labelmap_flag:
                        ignore_button = ignore_col2.button(
                            "Upload without Labelmap", key='ignore_labelmap')
                        if ignore_button:
                            continue_upload()
                            success = True
                    elif session_state.generate_labelmap_flag:
                        check_output_nodes(labelmap.dict, num_output_nodes)
                        model_upload.create_new_model_pipeline(
                            label_map_string=labelmap.label_map_string)
                        success = True

                elif model_upload.compatibility_flag == ModelCompatibility.Compatible:
                    # IF ALL REQUIREMENTS ARE MET
                    check_output_nodes(labelmap.dict, num_output_nodes)
                    model_upload.file_upload = uploaded_file

                    model_upload.create_new_model_pipeline()
                    success = True
                else:
                    st.error(f"Failed to create new model")
                    success = False
                    st.stop()

                logger.debug(f"{success = }")

                if success:
                    st.success("Model uploaded successfully, you may proceed to select it "
                               "in the model selection page.")
                    # clear out submit button if upload was successful
                    submit_btn_place.empty()

        submit_button_name = 'Submit' if model_upload.has_submitted == False else 'Update'
        # # TODO #72 Change to 'Update' when 'has_submitted' == True
        submit_button = submit_btn_place.button(
            label=submit_button_name, key="submit")
        if submit_button:
            model_upload_submit()

    # st.write("vars(model_upload)")
    # st.write(vars(model_upload))
    # st.write("session_state.generate_labelmap_flag")
    # st.write(session_state.generate_labelmap_flag)
    # st.write("vars(labelmap)")
    # st.write(vars(labelmap))


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        user_model_upload_page()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
