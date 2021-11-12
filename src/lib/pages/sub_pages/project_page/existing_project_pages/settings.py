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
from time import sleep
import pandas as pd
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state as session_state
from core.utils.helper import create_dataframe
from data_manager.data_table_component.data_table import data_table
from data_manager.dataset_management import Dataset
from pages.sub_pages.training_page import training_dashboard

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

from path_desc import chdir_root
from core.utils.log import logger  # logger
from data_manager.database_manager import init_connection
from project.project_management import ExistingProjectPagination, NewProject, ProjectPagination, SettingsPagination, ProjectPermission, Project, query_project_datasets, remove_project_dataset

from pages.sub_pages.dataset_page.new_dataset import new_dataset
from pages.sub_pages.labelling_page import labelling_dashboard

from annotation.annotation_management import reset_editor_page
from training.training_management import NewTraining, Training


def show_selection(df: pd.DataFrame, name_col: str = 'Name'):
    names = df.loc[df['id'].isin(session_state.data_table_selection),
                   name_col].tolist()
    st.markdown("**Selected:**")
    for i, n in enumerate(names, start=1):
        st.markdown(f"{i}. {n}")


def project():
    if session_state.project.desc:
        st.markdown("Project description:")
        st.markdown(f"{session_state.project.desc}")
        st.markdown("""___""")

    st.markdown("<h3 style='color: darkred;'>Danger Zone</h3>",
                unsafe_allow_html=True)

    if st.checkbox("❗ Delete project ", key='btn_delete',
                   help="Confirmation will be asked."):
        if st.button("Confirm deletion?", key='btn_confirm'):
            logger.info("Deleting project")
            project_id = session_state.project.id
            project_name = session_state.project.name
            Project.delete_project(project_id, project_name)

            for k in session_state:
                del session_state[k]

            session_state.project_pagination = ProjectPagination.Dashboard
            st.experimental_rerun()


def dataset():
    # similar code to the table in existing_project_dashboard
    if session_state.project.datasets:
        data_table_cols = [
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
        # df.fillna('-', inplace=True)
        st.write(df)

        # >>>> Dataset table placeholders
        dataset_table_col1, dataset_table_col2 = st.columns([3, 0.5])

        dataset_table_col1.write("## Current Project's Datasets")
        dataset_table_col2.write(
            f"### Total datasets: {len(session_state.project.datasets)}")
        st.markdown("""Select from the following datasets to either remove from our 
            project dataset selection, or delete the datasets and 
            associated projects (danger).""")

        records = df.to_dict(orient='records')
        # NOTE: data_table must have an 'id' column, not 'ID'
        selected_ids = data_table(records, data_table_cols, checkbox=True,
                                  key='data_table_selection')
        st.write(f"{selected_ids = }")

        # Get all projects associated with the project datasets of the current project
        dataset_ids = df['id'].tolist()
        project_datasets, column_names = query_project_datasets(dataset_ids)
        projects_df = pd.DataFrame(project_datasets, columns=column_names)

        # ***** Project dataset removal
        st.markdown("___")
        st.markdown("**Remove from project dataset selection**")
        st.markdown(
            "NOTE: This will also remove the associated annotations in the current project.")
        if st.checkbox("Remove project datasets", key='btn_remove_project_ds',
                       help="""This will remove the datasets from the current project, 
                        and also remove the associated annotations at the same time."""):
            if not selected_ids:
                st.warning("Please select at least a dataset first")
                st.stop()
            else:
                show_selection(df)
            if st.button("Confirm removal from project dataset?", key='btn_confirm'):
                for dataset_id in selected_ids:
                    remove_project_dataset(
                        session_state.project.id, dataset_id)
                session_state.project.refresh_project_details()
                session_state.existing_project_pagination = ExistingProjectPagination.Dashboard
                st.experimental_rerun()

        # ***** Dataset deletion
        st.markdown("___")
        st.markdown("""
        <h3 style='color: darkred; 
        text-decoration: underline'>
        Danger Zone
        </h3>
        """, unsafe_allow_html=True)
        st.markdown("## Datasets with associated projects")
        st.warning("""The following project datasets also have its associated projects,
            if you decided to remove any of these datasets, the associated projects will 
            also be removed to avoid complications.""")
        st.dataframe(projects_df)

        if st.checkbox("❗ Delete datasets", key='btn_delete_ds',
                       help="""WARNING: This will also delete all the associated projects!
                        Confirmation will be asked."""):
            if not selected_ids:
                st.warning("Please select at least a dataset first")
                st.stop()
            else:
                show_selection(df)
            if st.button("Confirm deletion of selected datasets and associated projects?", key='btn_confirm'):
                dataset_names = df.loc[df['id'].isin(
                    selected_ids), 'Name'].values.astype(str)
                for name in dataset_names:
                    logger.info(f"Deleting dataset of name: {name}")
                    Dataset.delete_dataset(name)

                project_ids_and_names = projects_df.loc[
                    projects_df['Dataset ID'].isin(
                        selected_ids), ['Project ID', 'Project Name']
                ].values.astype(str)
                for id, name in project_ids_and_names:
                    logger.info(f"Deleting project ID {id} of name: {name}")
                    Project.delete_project(id, name)

                for k in session_state:
                    del session_state[k]

                session_state.project_pagination = ProjectPagination.Dashboard
                st.experimental_rerun()
    else:
        st.warning("No dataset selected for this project yet! "
                   "Please add one by selecting from the options below.")


def training():
    pass


def models():
    pass


def index(RELEASE=True):
    logger.debug("At Exisiting Project Project INDEX")
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
        SettingsPagination.Project: project,
        SettingsPagination.Dataset: dataset,
        SettingsPagination.Training: training,
        SettingsPagination.Models: models
    }
    if 'settings_pagination' not in session_state:
        session_state.settings_pagination = SettingsPagination.Project

    logger.debug(
        f"Entering settings page: {session_state.settings_pagination}")

    session_state.append_project_flag = ProjectPermission.ViewOnly

    # >>>> Pagination RADIO >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    settings_page_options = [p.name for p in SettingsPagination]

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
