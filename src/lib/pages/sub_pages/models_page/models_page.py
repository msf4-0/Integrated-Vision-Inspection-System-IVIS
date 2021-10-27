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
from copy import deepcopy
from pathlib import Path
from time import sleep

import pandas as pd
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
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


# >>>> User-defined Modules >>>>
from core.utils.form_manager import remove_newline_trailing_whitespace
from core.utils.log import logger  # logger


from path_desc import chdir_root
from data_manager.database_manager import init_connection
from data_manager.data_table_component.data_table import data_table
from data_manager.database_manager import init_connection
from pages.sub_pages.models_page.models_subpages.user_model_upload import user_model_upload_page
from pages.sub_pages.training_page.new_training_subpages import new_training_augmentation_config, new_training_infodataset, new_training_training_config
from pages.sub_pages.training_page import run_training_page
from project.project_management import Project
from training.model_management import Model, ModelsPagination, NewModel, get_segmentation_model_funcs
from training.training_management import (NewTraining, NewTrainingPagination,
                                          NewTrainingSubmissionHandlers,
                                          Training)
from user.user_management import User
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration <<<<
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])
# <<<< Variable Declaration <<<<


def existing_models():
    logger.debug("[NAVIGATOR] At `models_page.py` `existing_models` function")

    if isinstance(session_state.new_training, NewTraining):
        # done creating the info in database, thus converting it
        # to a Training instance
        session_state.new_training = Training.from_new_training(
            session_state.new_training, session_state.project)
        logger.debug("Converted NewTraining to Training instance")

    existing_models, existing_models_column_names = deepcopy(Model.query_model_table(
        for_data_table=True,
        return_dict=True,
        deployment_type=session_state.new_training.deployment_type))

    # st.write(vars(session_state.training))

    # ************************* SESSION STATE ************************************************
    # if "existing_models_table" not in session_state:
    #     session_state.existing_models_table = None

    # ************************* SESSION STATE ************************************************

    # ************************ COLUMN PLACEHOLDER ********************************************
    st.write(f"**Step 1: Select a pretrained Deep Learning model "
             "from the table to be used for training, "
             "or you may also choose to upload a model:**")
    to_model_upload_page_button_place = st.empty()

    if session_state.project.deployment_type == "Image Classification":
        st.markdown("This table is obtained from Keras available pretrained models for image classification"
                    "[here](https://www.tensorflow.org/api_docs/python/tf/keras/applications).")
    elif session_state.project.deployment_type == "Object Detection with Bounding Boxes":
        st.markdown("This table is obtained from TensorFlow Object Detection Model Zoo "
                    "[here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). "
                    "The speed here actually means latency (ms), thus the lower the better; while higher COCO mAP score is better.")
    elif session_state.project.deployment_type == "Semantic Segmentation with Polygons":
        st.markdown("This table is obtained from `keras-unet-collection`'s GitHub repository "
                    "[here](https://github.com/yingkaisha/keras-unet-collection).")

    # - Removed the columns arrangement
    main_col1, main_col2 = st.columns([3, 1])
    # ************************ COLUMN PLACEHOLDER ********************************************

    def to_model_upload_page():
        session_state.models_pagination = ModelsPagination.ModelUpload

        if "existing_models_table" in session_state:
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
    # ******************************* DATA TABLE *******************************

    # CALLBACK >>>> Returns model_id -> store inside model class

    def instantiate_model():
        model_df_row = Model.filtered_models_dataframe(models=existing_models,
                                                       # care the name of the column is capitalized
                                                       dataframe_col="Name",
                                                       #    filter_value=session_state.existing_models_table[0],
                                                       filter_value=session_state.existing_models_table,
                                                       column_names=existing_models_column_names)

        # Instantiate Attached Model in `new_training` object
        session_state.new_training.attached_model = Model(
            model_row=model_df_row[0])

    # with main_col1.container():
        # not using `data_table` to avoid unnecessary complexity
        # data_table(rows=existing_models,
        #            columns=existing_models_columns,
        #            checkbox=False,
        #            key='existing_models_table', on_change=instantiate_model)

    logger.debug(f"""Loading pretrained model details for Project ID:
        {session_state.project.id} with deployment type:
        '{session_state.project.deployment_type}'""")
    if session_state.project.deployment_type == "Semantic Segmentation with Polygons":
        # we need the `model_func` column
        for_display = False
    else:
        for_display = True
    models_df = Model.get_pretrained_model_details(
        session_state.project.deployment_type,
        for_display=for_display
    )

    if session_state.new_training.attached_model is not None:
        # then this has already selected a model before
        model_name = session_state.new_training.attached_model.name
        model_idx = int(models_df.query(
            '`Model Name` == @model_name').index[0]
        )
    else:
        if session_state.project.deployment_type == "Object Detection with Bounding Boxes":
            # default to a good default SSD model to show to the user
            model_idx = 22
        elif session_state.project.deployment_type == "Image Classification":
            unwanted_idxs = models_df.loc[models_df['Model Name'].str.contains(
                "(deprecated|Mask R-CNN)")].index
            # drop deprecated models and Mask R-CNN model
            models_df = models_df.drop(unwanted_idxs).reset_index(drop=True)
            model_idx = int(models_df.loc[
                models_df['Model Name'] == "ResNet50"].index[0])
        else:
            # get only the models used for our training in this app
            model_func_names = list(get_segmentation_model_funcs().keys())
            models_df = models_df[models_df['model_func'].isin(
                model_func_names)]
            model_idx = int(models_df.loc[
                models_df['Model Name'] == 'U-net'].index[0])
            models_df.drop(columns='model_func', inplace=True)

    st.dataframe(models_df, width=1500)

    model_selected = st.selectbox(
        "Please select a Pre-trained model",
        options=models_df['Model Name'],
        index=model_idx,
        key="existing_models_table",
    )

    # must do it this way instead of using callback on selectbox,
    # to be able to show the model info form on the first render
    if model_selected:
        instantiate_model()

    # ******************************* DATA TABLE *******************************

    # >>>> MAIN_COL2

    if session_state.new_training.attached_model:
        # with main_col2.container():

        if session_state.new_training.has_submitted[NewTrainingPagination.Model]:
            current_name = session_state.new_training.training_model.name
            current_desc = session_state.new_training.training_model.desc
            st.info(f"""
            **Current Model Name**: {current_name}  \n
            **Current Description**: {current_desc}  \n
            """)

        # >>>> GET PERF METRICS of SELECTED MODEL
        y = session_state.new_training.attached_model.get_perf_metrics()

        df_metrics = Model.create_perf_metrics_table(y)

        # - can consider adding the bottom line after done creating self.model_input_size
        # Model Input Size: {session_state.new_training.attached_model.model_input_size}
        model_information = f"""
        ## Model Information:
        ### Name: 
        {session_state.new_training.attached_model.name}
        ### Framework: 
        {session_state.new_training.attached_model.framework}
        """
        st.info(model_information)
        if not df_metrics.empty:
            st.write(f"#### Metrics:")
            st.table(df_metrics)
        session_state.new_training_place["attached_model_selection"] = st.empty(
        )
        # with main_col1.container():
        # CALLBACK: CHECK IF MODEL NAME EXISTS

        def check_if_name_exist(field_placeholder, conn):
            context = {'column_name': 'name',
                       'value': session_state.training_model_name}

            if session_state.training_model_name:
                if session_state.new_training.training_model.check_if_exists(context, conn):
                    session_state.new_training.training_model.name = ''
                    field_placeholder['training_model_name'].error(
                        f"Model name used. Please enter a new name")
                    sleep(0.7)
                    field_placeholder['training_model_name'].empty()
                    logger.error(
                        f"Training Model name used. Please enter a new name")
                else:
                    session_state.new_training.training_model.name = session_state.training_model_name
                    logger.info(
                        f"Training Model name fresh and ready to rumble {session_state.new_training.training_model.name}")
            else:
                pass

        # ******************************** MODEL TITLE ********************************
        st.write(
            f"**Step 2: Enter the name of your model:** ")

        st.text_input('Model Name', key="training_model_name",
                      value=session_state.new_training.training_model.name,
                      help="Enter the name of the model to be exported after training",
                      on_change=check_if_name_exist,
                      args=(session_state.new_training_place, conn,))
        session_state.new_training_place['training_model_name'] = st.empty(
        )
        st.write(session_state.training_model_name)

        # ************************* MODEL DESCRIPTION (OPTIONAL) *************************
        description = st.text_area(
            "Description (Optional)", key="training_model_desc",
            value=session_state.new_training.training_model.desc,
            help="Enter the description of the model")

        if description:
            session_state.new_training.training_model.desc = remove_newline_trailing_whitespace(
                description)
        else:
            pass


def index():

    chdir_root()  # change to root directory
    RELEASE = True

    # ****************** TEST ******************************
    # For testing:
    # Instantiate Project Class and pass name as argument
    # Instiate New Training Class and set name and ID

    if not RELEASE:
        logger.debug("At Models INDEX")

        # ************************TO REMOVE************************
        with st.sidebar.container():
            st.image("resources/MSF-logo.gif", use_column_width=True)
            st.title("Integrated Vision Inspection System", anchor='title')
            st.header(
                "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
            st.markdown("""___""")

        # ************************TO REMOVE************************
        project_id_tmp = 4
        training_id_tmp = 2
        logger.debug(f"Entering Project {project_id_tmp}")

        # session_state.append_project_flag = ProjectPermission.ViewOnly

        if "project" not in session_state:
            session_state.project = Project(project_id_tmp)
            logger.debug("Inside")
        if 'user' not in session_state:
            session_state.user = User(1)
        if "new_training_pagination" not in session_state:
            session_state.new_training_pagination = NewTrainingPagination.Model

        if 'new_training' not in session_state:

            session_state.new_training = NewTraining(training_id_tmp,
                                                     project=session_state.project)
            session_state.new_training.name = "My Tenth Training"
            session_state.new_training.deployment_type = "Object Detection with Bounding Boxes"

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
        # old page before this models_page
        # ModelsPagination.TrainingInfoDataset: new_training_infodataset.infodataset,
        # the current page function in this script
        ModelsPagination.ExistingModels: existing_models,
        ModelsPagination.ModelUpload: user_model_upload_page,

        # ModelsPagination.TrainingConfig: new_training_training_config.training_configuration,
        # ModelsPagination.AugmentationConfig: new_training_augmentation_config.augmentation_configuration,
        # ModelsPagination.Training: run_training_page.index,
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

    logger.debug(
        f"Entering Models Page: {session_state.models_pagination}")

    if session_state.models_pagination == ModelsPagination.ExistingModels:
        st.write(f"### Models Selection")
    # >>>> MAIN FUNCTION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    models_page[session_state.models_pagination]()
    # ************************* NEW TRAINING SECTION PAGINATION BUTTONS **********************
    # Placeholder for Back and Next button for page navigation
    # new_training_section_back_button_place, _, \
    #     new_training_section_next_button_place = st.columns([0.5, 3, 0.5])
    new_training_section_next_button_place = st.empty()

    # typing.NamedTuple type

    new_training_model_submission_dict = NewTrainingSubmissionHandlers(
        insert=session_state.new_training.training_model.create_new_project_model_pipeline,
        update=session_state.new_training.training_model.update_new_project_model_pipeline,
        context={
            'training_model_name': session_state.new_training.training_model.name,
            'attached_model_selection': session_state.existing_models_table
        },
        name_key='training_model_name'
    )

    # *************************************CALLBACK: NEXT BUTTON *************************************
    def to_training_configuration_page():

        # run submission according to current page
        # NEXT page if all constraints met

        # >>>> IF IT IS A NEW SUBMISSION
        if not session_state.new_training.has_submitted[NewTrainingPagination.Model]:

            if session_state.new_training.training_model.check_if_field_empty(
                    context=new_training_model_submission_dict.context,
                    field_placeholder=session_state.new_training_place,
                    name_key=new_training_model_submission_dict.name_key):
                # run create new project model pipeline
                if new_training_model_submission_dict.insert(
                        attached_model=session_state.new_training.attached_model,
                        project_name=session_state.project.name,
                        training_name=session_state.new_training.name,
                        training_id=session_state.new_training.id):

                    # run update training attached model
                    if session_state.new_training.update_training_attached_model(
                            attached_model_id=session_state.new_training.attached_model.id,
                            training_model_id=session_state.new_training.training_model.id):
                        # set has_submitted =True
                        session_state.new_training.has_submitted[NewTrainingPagination.Model] = True
                        logger.info(
                            f"Successfully created new training model {session_state.new_training.training_model.id}")

                        session_state.new_training_pagination = NewTrainingPagination.TrainingConfig

        elif session_state.new_training.has_submitted[NewTrainingPagination.Model]:
            if session_state.new_training.training_model.name:
                # UPDATE Database
                # Training Name,Desc, Dataset chosen, Partition Size
                if new_training_model_submission_dict.update(
                        attached_model=session_state.new_training.attached_model,
                        project_name=session_state.project.name,
                        training_name=session_state.new_training.name):
                    # run update training attached model
                    if session_state.new_training.update_training_attached_model(
                            attached_model_id=session_state.new_training.attached_model.id,
                            training_model_id=session_state.new_training.training_model.id):
                        logger.info(
                            f"Successfully updated new training model {session_state.new_training.training_model.id}")
                        # session_state.new_training_pagination = NewTrainingPagination.TrainingConfig
                        for page, submitted in session_state.new_training.has_submitted.items():
                            if not submitted:
                                session_state.new_training_pagination = page
                                break
                        else:
                            # go to Training page if all forms have been submitted
                            session_state.new_training_pagination = NewTrainingPagination.Training
                        logger.debug('New Training Pagination: '
                                     f'{session_state.new_training_pagination}')
            else:
                session_state.new_training_place['training_model_name'].error(
                    'Training Model Name already exists, please enter a new name')

    # ***************** NEXT BUTTON **************************
    with new_training_section_next_button_place:
        st.button("Submit Model Info", key="models_page_next_button",
                  on_click=to_training_configuration_page)

    # ***************** BACK BUTTON **************************
    def to_training_infodataset_page():
        # session_state.models_pagination = ModelsPagination.TrainingInfoDataset
        # must update this pagination variable too to make things work properly
        session_state.new_training_pagination = NewTrainingPagination.InfoDataset

    # with new_training_section_back_button_place:
    st.sidebar.button("Back to Modify Training Info", key="models_page_back_button",
                      on_click=to_training_infodataset_page)

    st.write("vars(session_state.new_training)")
    st.write(vars(session_state.new_training))
    st.write("vars(session_state.new_training.training_model)")
    st.write(vars(session_state.new_training.training_model))
    st.write("vars(session_state.new_training.attached_model)")
    st.write(vars(session_state.new_training.attached_model))


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        index()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
