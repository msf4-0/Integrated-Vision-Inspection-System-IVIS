""" 
Title: Settings Page
Date: 12/11/2021 
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
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>
# DEFINE Web APP page configuration
# layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
# LIB_PATH = SRC / "lib"
# if str(LIB_PATH) not in sys.path:
#     sys.path.insert(0, str(LIB_PATH))  # ./lib
# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

from core.utils.log import logger  # logger
from core.utils.helper import create_dataframe
from data_manager.data_table_component.data_table import data_table
from data_manager.dataset_management import Dataset
from training.model_management import Model, query_current_project_models, query_uploaded_models
from user.user_management import UserRole
from project.project_management import (ExistingProjectPagination, SettingsPagination,
                                        ProjectPermission, Project,
                                        query_project_datasets, remove_project_dataset)
from training.training_management import Training


def danger_zone_header():
    st.markdown("""
    <h3 style='color: darkred; 
    text-decoration: underline'>
    Danger Zone
    </h3>
    """, unsafe_allow_html=True)


def show_selection(df: pd.DataFrame, selected_ids: List[int], name_col: str = 'Name'):
    names = df.loc[df['id'].isin(selected_ids),
                   name_col].tolist()
    st.markdown("**Selected:**")
    for i, n in enumerate(names, start=1):
        st.markdown(f"{i}. {n}")


def project():
    danger_zone_header()
    st.markdown("""**WARNING**: This will delete all the information associated with 
        this project. Confirmation will be asked.""")
    if st.checkbox("❗ Delete project ", key='cbox_delete_project'):
        if st.button("Confirm deletion?", key='btn_confirm_delete_project'):
            logger.info("Deleting project")
            project_id = session_state.project.id
            Project.delete_project(project_id)

            # reset all session_states
            session_state.clear()

            session_state.existing_project_pagination = ExistingProjectPagination.Dashboard
            st.experimental_rerun()


def dataset():
    if not session_state.project.datasets:
        # to handle situation without any dataset uploaded yet
        st.warning("No dataset has been selected for this project yet. "
                   "Please go back to relevant pages to create a new dataset first.")
        st.stop()

    # similar code to the table in existing_project_dashboard
    DATA_TABLE_COLS = [
        {
            'field': "id",
            'headerName': "ID",
            'headerAlign': "center",
            'align': "center",
            'flex': 1,
            'hideSortIcons': True,
        },
        {
            'field': "Name",
            'headerName': "Name",
            'headerAlign': "center",
            'align': "center",
            'flex': 5,
            'hideSortIcons': True,
        },
        {
            'field': "Description",
            'headerName': "Description",
            'headerAlign': "center",
            'align': "center",
            'flex': 5,
            'hideSortIcons': True,
        },
        {
            'field': "Dataset Size",
            'headerName': "Dataset Size",
            'headerAlign': "center",
            'align': "center",
            'flex': 3,
            'hideSortIcons': True,
        },
        {
            'field': "Date/Time",
            'headerName': "Date/Time",
            'headerAlign': "center",
            'align': "center",
            'flex': 3,
            'hideSortIcons': True,
        },
    ]
    df = create_dataframe(session_state.project.datasets,
                          column_names=session_state.project.column_names,
                          sort=True, sort_by='ID', date_time_format=True)
    df.drop(columns='File Type', inplace=True)
    df.rename(columns={'ID': 'id'}, inplace=True)
    df['Date/Time'] = df['Date/Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # >>>> Dataset table placeholders
    dataset_table_col1, dataset_table_col2 = st.columns([3, 0.5])

    dataset_table_col1.write("## Current Project's Datasets")
    dataset_table_col2.write(
        f"### Total datasets: {len(session_state.project.datasets)}")
    st.markdown("""Select from the following datasets to either remove from the current
        project's dataset selection, or delete the datasets and annotations of associated
        projects (danger).""")

    records = df.to_dict(orient='records')
    selected_ids = data_table(records, DATA_TABLE_COLS, checkbox=True,
                              key='data_table_dataset')
    # st.write(f"{selected_ids = }")

    # Get all projects associated with the project datasets of the current project
    dataset_ids = df['id'].tolist()
    project_datasets, column_names = query_project_datasets(dataset_ids)
    projects_df = pd.DataFrame(project_datasets, columns=column_names)

    # ***** Project dataset removal
    st.markdown("___")
    st.markdown("**Remove from project dataset selection**")
    st.markdown("""**NOTE**: This will not permanently delete the datasets. This will
        only remove the dataset from the **current project** and also the **associated
        annotations** in the current project.""")
    if st.checkbox("Remove selected project datasets", key='cbox_remove_project_ds',
                   help="""This will remove the datasets from the current project, 
                    and also remove the associated annotations at the same time."""):
        if not selected_ids:
            st.warning("Please select at least a dataset first")
            st.stop()
        else:
            show_selection(df, selected_ids)
        if st.button("Confirm removal from project dataset?",
                     key='btn_confirm_remove_project_ds'):
            for dataset_id in selected_ids:
                remove_project_dataset(
                    session_state.project.id, dataset_id)
            session_state.project.refresh_project_details()
            if not session_state.project.datasets:
                session_state.existing_project_pagination = ExistingProjectPagination.Dashboard
            st.experimental_rerun()

    # ***** Dataset deletion
    st.markdown("___")
    danger_zone_header()
    st.markdown("## Datasets with associated projects")
    st.dataframe(projects_df)

    if st.checkbox("❗ Delete selected datasets", key='cbox_delete_ds',
                   help="""WARNING: This will also delete all the annotations of 
                    the associated projects! Confirmation will be asked."""):
        if not selected_ids:
            st.warning("Please select at least a dataset first")
            st.stop()
        st.warning("""**WARNING**: The following project datasets also have its 
            associated projects (including the current project), if you decided to 
            delete any of these datasets, the labeled annotations of the associated 
            projects will also be deleted.""")
        show_selection(df, selected_ids)
        if st.button("Confirm deletion of selected datasets and associated annotations?",
                     key='btn_confirm_delete_ds'):
            for id in selected_ids:
                logger.info(f"Deleting dataset of ID: {id}")
                Dataset.delete_dataset(id)

            session_state.project.refresh_project_details()
            if not session_state.project.datasets:
                session_state.existing_project_pagination = ExistingProjectPagination.Dashboard
            st.experimental_rerun()


def training():
    DEPLOYMENT_TYPE = session_state.project.deployment_type
    PROGRESS_COLUMN_HEADER = {
        "Image Classification": 'Steps',
        "Object Detection with Bounding Boxes": 'Checkpoint / Steps',
        "Semantic Segmentation with Polygons": 'Checkpoint / Steps'
    }
    DATA_TABLE_COLS = [
        {
            'field': "id",
            'headerName': "ID",
            'headerAlign': "center",
            'align': "center",
            'flex': 40,
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
            'headerName': "Project Model Name",
            'headerAlign': "center",
            'align': "center",
            'flex': 120,
            'hideSortIcons': True,
        },
        {
            'field': "Base Model Name",
            'headerName': "Pre-trained Model",
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
            'headerName': f"{PROGRESS_COLUMN_HEADER[DEPLOYMENT_TYPE]}",
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

    # similar stuff with deployment's model_selection page
    options = ('Project Training Sessions & Models', 'User-Uploaded Models')
    selected_type = st.radio(
        'Select the type of data to check', options,
        key='selected_type',
        help='Project Model is a model trained in our application.')

    if selected_type == 'Project Training Sessions & Models':
        st.markdown("## All Training Sessions for Current Project")
        all_project_training, project_training_column_names = \
            Training.query_all_project_training(session_state.project.id,
                                                deployment_type=DEPLOYMENT_TYPE,
                                                return_dict=True,
                                                for_data_table=True,
                                                progress_preprocessing=True)
        if not all_project_training:
            st.info("""No training sessions created for this project yet.""")
            st.stop()

        data = all_project_training
        # need this to get the IDs and names later
        model_id_col_name = "Model ID"
        model_name_col_name = "Model Name"
    else:
        st.markdown(f"## All User-Uploaded Models for {DEPLOYMENT_TYPE}")
        models, _ = query_uploaded_models(
            for_data_table=True,
            return_dict=True,
            deployment_type=DEPLOYMENT_TYPE)
        if not models:
            st.info("""No uploaded models available for this deployment type.""")
            st.stop()

        data = models
        model_id_col_name = "id"
        model_name_col_name = "Name"

    unique_key = selected_type.split()[0]
    columns = DATA_TABLE_COLS if selected_type == 'Project Training Sessions & Models' else UPLOADED_DT_COLS
    # this ID is Training ID for Project Model, but Model ID for User-uploaded Model
    selected_ids = data_table(
        data, columns, checkbox=True, key=f'data_table_model_selection_{unique_key}')

    st.markdown("___")
    danger_zone_header()

    if selected_type == 'Project Training Sessions & Models':
        st.markdown(
            "## Select training sessions from the table above to delete")
        if st.checkbox("Delete selected training sessions", key='cbox_delete_training',
                       help="""This will delete both the selected training sessions
                    **and also the associated models**"""):
            if not selected_ids:
                st.warning("Please select at least a training session first")
                st.stop()
            st.warning("""**WARNING**: This will **also delete** the models associated 
                with the training sessions.""")
            # use set for faster membership filtering
            selected_ids = set(selected_ids)
            selected_records = filter(lambda x: x['id'] in selected_ids,
                                      all_project_training)
            st.markdown("**Selected training sessions:**")
            for i, rec in enumerate(selected_records, start=1):
                training_name = rec['Training Name']
                st.markdown(f"{i}. {training_name}")

            if st.button("Confirm deletion of selected training sessions and associated models?",
                         key='btn_confirm_delete_training'):
                for t_id in selected_ids:
                    Training.delete_training(t_id)
                st.experimental_rerun()

        st.markdown("___")
        st.markdown("### Or just delete the selected models")
        st.markdown("""**NOTE**: This will delete the project models together with
            all the associated information.""")
    else:
        st.markdown("### Delete user-uploaded models")
        st.markdown("This will delete all the data related to the selected user-uploaded "
                    "models, including both the database data and the saved model files.")

    if st.checkbox("Delete selected models", key='cbox_delete_model',
                   help="""This will delete the selected models"""):
        if not selected_ids:
            st.warning("Please select at least a model first")
            st.stop()

        selected_ids = set(selected_ids)
        selected_records = filter(lambda x: x['id'] in selected_ids, data)

        st.markdown("**Selected models:**")
        model_ids = []
        for i, rec in enumerate(selected_records, start=1):
            model_ids.append(rec[model_id_col_name])
            model_name = rec[model_name_col_name]
            st.markdown(f"{i}. {model_name}")

        if st.button("Confirm deletion of selected models?",
                     key='btn_confirm_delete_model'):
            for model_id in model_ids:
                Model.delete_model(model_id)
            st.experimental_rerun()


def models():
    st.markdown("NOT USING THIS FOR NOW")
    st.stop()
    DATA_TABLE_COLS = [
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
    project_models, _ = query_current_project_models(
        project_id=session_state.project.id,
        for_data_table=True,
        return_dict=True)

    if not project_models:
        st.info("""No model has been created for this project yet.""")
        st.stop()

    st.write(pd.DataFrame(project_models, columns=_))
    st.write(project_models)

    st.markdown("## All Project Models of the Current Project")
    selected_ids = data_table(project_models, DATA_TABLE_COLS,
                              checkbox=True, key='data_table_training')

    danger_zone_header()
    st.markdown("""NOTE: This will also remove the associations between the project 
        models and the training sessions.""")
    if st.checkbox("Delete selected models", key='cbox_delete_training',
                   help="""This will delete the selected models"""):
        if not selected_ids:
            st.warning("Please select at least a model first")
            st.stop()
        else:
            # use set for faster membership filtering
            selected_ids = set(selected_ids)
            selected_records = filter(lambda x: x['id'] in selected_ids,
                                      project_models)
            st.markdown("**Selected:**")
            for i, rec in enumerate(selected_records, start=1):
                training_name = rec['Training Name']
                st.markdown(f"{i}. {training_name}")

        if st.button("Confirm deletion of selected models?",
                     key='btn_confirm_delete_training'):
            for t_id in selected_ids:
                Training.delete_training(t_id)
            st.experimental_rerun()


def index(RELEASE=True):
    logger.debug("At Settings Page")

    if session_state.user.role == UserRole.Annotator:
        st.warning(
            "You are not allowed to access to settings page to delete any data.")
        st.stop()

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
        project_id_tmp = 7
        logger.debug(f"Entering Project {project_id_tmp}")

        session_state.append_project_flag = ProjectPermission.ViewOnly

        if "project" not in session_state:
            session_state.project = Project(project_id_tmp)
            logger.debug("Inside")

    # ************************ Settings PAGINATION *************************
    settings_page2func = {
        SettingsPagination.TrainingAndModel: training,
        SettingsPagination.Dataset: dataset,
        SettingsPagination.Project: project,
        # not using this for now
        # SettingsPagination.Models: models
    }
    if 'settings_pagination' not in session_state:
        session_state.settings_pagination = SettingsPagination.TrainingAndModel

    logger.debug(
        f"Entering settings page: {session_state.settings_pagination}")

    session_state.append_project_flag = ProjectPermission.ViewOnly

    # >>>> Pagination RADIO >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    settings_page_options = (
        'Training/Model Deletion', 'Dataset Deletion', 'Project Deletion')

    # >>>> CALLBACK for RADIO >>>>
    def settings_page_navigator():
        navigation_selected = session_state.settings_page_navigator_radio
        navigation_selected_idx = settings_page_options.index(
            navigation_selected)
        # IntEnum can be compared with int
        session_state.settings_pagination = navigation_selected_idx

    st.sidebar.radio("Settings navigation", options=settings_page_options,
                     index=session_state.settings_pagination,
                     on_change=settings_page_navigator, key="settings_page_navigator_radio")
    st.sidebar.markdown("___")

    # >>>> MAIN FUNCTION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    settings_page2func[session_state.settings_pagination]()


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        # False for debugging
        index(RELEASE=False)
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
