""" 
Title: Deployment - model selection page
Date: 16/11/2021
Author: Anson Tan Chen Tung
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
from pathlib import Path
import shutil
import sys
from typing import List

import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state

# >>>> **************** TEMP (for debugging) **************** >>>
# DEFINE Web APP page configuration for debugging on this page
# layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
# LIB_PATH = SRC / "lib"
# if str(LIB_PATH) not in sys.path:
#     sys.path.insert(0, str(LIB_PATH))  # ./lib

# >>>> **************** TEMP **************** >>>

# >>>> User-defined Modules >>>>
from core.utils.log import logger
from deployment.deployment_management import Deployment, DeploymentPagination
from machine_learning.trainer import Trainer
from machine_learning.command_utils import run_tensorboard
from machine_learning.utils import load_labelmap
from machine_learning.visuals import pretty_format_param
from training.labelmap_generator import labelmap_generator
from training.labelmap_management import Labels
from user.user_management import User, UserRole
from project.project_management import Project
from training.training_management import Training
from training.model_management import Model, query_current_project_models, query_uploaded_models
from data_manager.data_table_component.data_table import data_table


def create_labelmap_file(class_names: List[str], output_dir: Path, deployment_type: str):
    """`output_dir` is the directory to store the `labelmap.pbtxt` file"""
    labelmap_string = Labels.generate_labelmap_string(
        class_names,
        framework='TensorFlow',
        deployment_type=deployment_type)
    Labels.generate_labelmap_file(
        labelmap_string=labelmap_string,
        dst=output_dir,
        framework='TensorFlow',
        deployment_type=deployment_type)


def index(RELEASE=True):
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
        # for Anson: 4 for TFOD, 9 for img classif, 30 for segmentation
        # uploaded pet segmentation: 96
        project_id_tmp = 9
        logger.debug(f"Entering Project {project_id_tmp}")
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
        new_training_progress_bar = st.progress(0)
        new_training_progress_bar.progress(2 / 4)
    # ****************** TEST END ******************************

    project: Project = session_state.project
    DEPLOYMENT_TYPE: str = project.deployment_type
    user: User = session_state.user
    if user.role > UserRole.Developer1:
        st.warning("You are not allowed to deploy model.")
        st.stop()

    PROJECT_DT_COLS = [
        {
            'field': "Model ID",
            'headerName': "ID",
            'headerAlign': "center",
            'align': "center",
            'flex': 5,
            'hideSortIcons': True,
        },
        {
            'field': "Name",
            'headerName': "Name",
            'headerAlign': "center",
            'align': "center",
            'flex': 15,
            'hideSortIcons': True,
        },
        {
            'field': "Base Model Name",
            'headerName': "Pre-trained Model",
            'headerAlign': "center",
            'align': "center",
            'flex': 15,
            'hideSortIcons': True,
        },
        {
            'field': "Description",
            'headerName': "Description",
            'headerAlign': "center",
            'align': "center",
            'flex': 15,
            'hideSortIcons': True,
        },
        {
            'field': "Metrics",
            'headerName': "Metrics",
            'headerAlign': "center",
            'align': "center",
            'flex': 20,
            'hideSortIcons': True,
        },
        {
            'field': "Date/Time",
            'headerAlign': "center",
            'align': "center",
            'flex': 10,
            'hideSortIcons': True,
            'type': 'date',
        },
    ]

    # the only difference is without "Base Model Name" and "Metrics" columns
    UPLOADED_DT_COLS = [
        {
            'field': "id",
            'headerName': "Model ID",
            'headerAlign': "center",
            'align': "center",
            'flex': 5,
            'hideSortIcons': True,
        },
        {
            'field': "Name",
            'headerName': "Name",
            'headerAlign': "center",
            'align': "center",
            'flex': 15,
            'hideSortIcons': True,
        },
        {
            'field': "Description",
            'headerName': "Description",
            'headerAlign': "center",
            'align': "center",
            'flex': 15,
            'hideSortIcons': True,
        },
        {
            'field': "Date/Time",
            'headerAlign': "center",
            'align': "center",
            'flex': 10,
            'hideSortIcons': True,
            'type': 'date',
        },
    ]

    st.header(f"Model Selection for Deployment:")
    options = ('Project Model', 'User-Uploaded Model')
    selected_model_type = st.radio(
        'Select the type of model', options,
        key='selected_model_type',
        help='Project Model is a model trained in our application.')

    if selected_model_type == 'Project Model':
        st.markdown("## All Trained Project Models for Current Project")
        # namedtuple of query from DB
        models, _ = query_current_project_models(
            project.id,
            for_data_table=True,
            return_dict=True,
            prettify_metrics=True,
            trained=True)
        if not models:
            st.info("""No trained models available for this project.""")
            st.stop()
    else:
        st.markdown(f"## All User-Uploaded Models for {DEPLOYMENT_TYPE}")
        models, _ = query_uploaded_models(
            for_data_table=True,
            return_dict=True,
            deployment_type=DEPLOYMENT_TYPE)
        if not models:
            st.info("""No uploaded models available for this deployment type.""")
            st.stop()

    st.markdown("Select one to display more information about the trained model")

    def reset_cache():
        logger.debug("Clearing cache")
        # need to clear loaded model's cache
        st.legacy_caching.clear_cache()
    unique_key = selected_model_type.split()[0]
    columns = PROJECT_DT_COLS if selected_model_type == 'Project Model' else UPLOADED_DT_COLS
    # this ID is Training ID for Project Model, but Model ID for User-uploaded Model
    selected_id = data_table(
        models, columns, checkbox=False,
        key=f'data_table_model_selection_{unique_key}',
        on_change=reset_cache)

    deploy_button_col, _ = st.columns(2)

    if not selected_id:
        st.stop()
        # take the first one if none is selected
        # selected_id = models[0]['id']
    else:
        # index into the single value in the List[int]
        selected_id = selected_id[0]
    selected_row = next(m for m in models
                        if m['id'] == selected_id)

    model_info_col, metric_col = st.columns(2)

    if selected_model_type == 'Project Model':
        # INITIALIZE training instance here for information with the model
        training = Training(selected_id, project)
        trained_model = training.training_model
        # store the trainer to use for training, inference and deployment
        with st.spinner("Initializing trainer ..."):
            logger.info("Initializing trainer")
            trainer = Trainer(project, training)
        # get all the training_path
        training_path = trainer.training_path

        model_information = f"""
        #### Name:
        {trained_model.name}
        #### Description:
        {selected_row['Description']}
        #### Pre-trained Model:
        {selected_row['Base Model Name']}
        """
        # Framework:
        # {selected_model.framework}
        with model_info_col:
            st.subheader("Selected Model Information:")
            st.info(model_information)

        with metric_col:
            metrics = pretty_format_param(trained_model.metrics)
            st.subheader("Final Metrics:")
            progress_text = pretty_format_param(
                training.progress, st_newlines=False, bold_name=True)
            st.markdown(f"Latest progress at {progress_text}")
            st.info(metrics)

        show_tb = st.button("📈 Show TensorBoard", key='btn_show_tensorboard')
        if show_tb:
            with st.spinner("Loading Tensorboard ..."):
                logdir = training_path['tensorboard_logdir']
                run_tensorboard(logdir)

        labelmap_path = training_path['labelmap_file']
        if not labelmap_path.exists():
            logger.debug(
                "Creating a new labelmap file because labelmap file was not created "
                "during training, due to the old training pipeline")
            create_labelmap_file(
                trainer.class_names,
                labelmap_path.parent, trainer.deployment_type)

        category_index = load_labelmap(labelmap_path)
        st.markdown("#### Categories loaded from labelmap file:")
        st.json(category_index)

        # If debugging, consider commenting the evaluation part for faster loading times
        st.markdown("___")
        st.header("Evaluation results:")
        st.markdown("""WARNING: If you have just deployed a model and it's not ended yet, 
        clicking this will reset the deployment process because evaluation on this model
        requires loading a new model and it requires resources.""")
        if st.checkbox("Show evaluation results", key='show_eval_res'):
            if 'deployment' in session_state:
                Deployment.reset_deployment_page()
                st.experimental_rerun()
            labelstudio_json_path = project.get_project_json_path()
            if not labelstudio_json_path.exists():
                with st.spinner("Exporting labeled data for evaluation ..."):
                    logger.info("Exporting tasks for evaluation ...")
                    project.export_tasks(for_training_id=training.id)
            with st.spinner("Running evaluation ..."):
                try:
                    trainer.evaluate()
                except Exception as e:
                    # uncomment this line to check the Traceback
                    # st.exception(e)
                    st.error("Some error has occurred. Please try "
                             "training/exporting the model again.")
                    logger.error(f"Error evaluating: {e}")
    else:
        model = Model(selected_id)
        uploaded_model_dir = model.get_path()
        logger.debug(f"{uploaded_model_dir = }")
        labelmap_paths = list(uploaded_model_dir.rglob("*.pbtxt"))
        if labelmap_paths:
            # should have only one file
            labelmap_path = labelmap_paths[0]
            try:
                category_index = load_labelmap(labelmap_path)
                # classes = [d['name'] for d in category_index.values()]
            except Exception as e:
                st.error(f"Error loading the uploaded labelmap file associated with "
                         "the model, cannot proceed to deployment without labelmap.")
                logger.error(f"Error loading the uploaded labelmap file: {e}")
                st.stop()
        else:
            st.error("This uploaded model does not include a **'labelmap.pbtxt'** file, "
                     "thus not compatible to be instantly deployed. You can generate a "
                     "new labelmap file if you are sure about the class labels associated "
                     "with the uploaded model.")
            labelmap_col, _ = st.columns(2)
            with labelmap_col:
                generate_labelmap_flag, label_map_string = labelmap_generator(
                    framework='TensorFlow',
                    deployment_type=DEPLOYMENT_TYPE)
            if not generate_labelmap_flag:
                st.stop()
            else:
                if st.button("Generate file", key='btn_generate_labelmap'):
                    if not label_map_string:
                        st.error("Please enter class names first!")
                        st.stop()
                    Labels.generate_labelmap_file(
                        label_map_string,
                        dst=uploaded_model_dir,
                        framework='TensorFlow', deployment_type=DEPLOYMENT_TYPE)
                    st.experimental_rerun()
                else:
                    st.stop()
        model_information = f"""
        #### Name:
        {selected_row['Name']}
        #### Description:
        {selected_row['Description']}
        """
        with model_info_col:
            st.info(model_information)
        if labelmap_paths:
            st.markdown("#### Categories loaded from uploaded labelmap file:")
            st.json(category_index)
        # Framework:
        # {selected_model.framework}

    # BUTTON to deploy the selected model
    def enter_deployment():
        Deployment.reset_deployment_page()
        st.legacy_caching.clear_cache()

        if selected_model_type == 'Project Model':
            export_path = project.get_export_path()
            if export_path.exists():
                # not required after showing evaluation to the user
                shutil.rmtree(export_path)
            session_state.deployment = Deployment.from_trainer(trainer)
        else:
            session_state.deployment = Deployment.from_uploaded_model(
                model, uploaded_model_dir, category_index)

        with deploy_button_col:
            with st.spinner("Preparing model for deployment ..."):
                session_state.deployment.run_preparation_pipeline()
        session_state.deployment_pagination = DeploymentPagination.Deployment

    with deploy_button_col:
        st.button("🛠️ Deploy selected model",
                  key='btn_deploy_selected_model',
                  on_click=enter_deployment)


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        # False for debugging
        index(RELEASE=False)
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
