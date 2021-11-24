"""
Title: Existing Project Dashboard
Date: 19/8/2021
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

from enum import IntEnum
from time import sleep
import sys
from pathlib import Path
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[5]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

from path_desc import chdir_root
from core.utils.log import logger
from core.utils.helper import create_dataframe, get_df_row_highlight_color, get_textColor, current_page, non_current_page
from core.utils.form_manager import remove_newline_trailing_whitespace
from user.user_management import User
from data_manager.database_manager import init_connection
from data_manager.dataset_management import NewDataset, query_dataset_list, get_dataset_name_list
from project.project_management import ProjectDashboardPagination, ProjectPermission, Project
from data_editor.editor_management import Editor
from data_editor.editor_config import editor_config
from pages.sub_pages.dataset_page.new_dataset import new_dataset

# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>

chdir_root()  # change to root directory


def dashboard(RELEASE=True, **kwargs):
    st.write(f"## **Overview:**")
    # TODO #79 Add dashboard to show types of labels and number of datasets
    # >>>>>>>>>>PANDAS DATAFRAME for LABEL DETAILS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    label_count_dict = session_state.project.get_existing_unique_labels(
        return_counts=True)
    df = session_state.project.editor.create_table_of_labels(label_count_dict)
    df.index.name = 'No.'
    df['Percentile (%)'] = df['Percentile (%)'].map("{:.2f}".format)
    styler = df.style

    # >>>> Annotation table placeholders
    annotation_col1, annotation_col2 = st.columns([3, 0.5])

    annotation_col1.write("### **Annotations**")
    annotation_col2.write(
        f"### Total labels: {len(session_state.project.editor.labels_results)}")

    st.table(styler.set_properties(**{'text-align': 'center'}).set_table_styles(
        [dict(selector='th', props=[('text-align', 'center')])]))

    # >>>>>>>>>>PANDAS DATAFRAME for LABEL DETAILS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # >>>>>>>>>>PANDAS DATAFRAME for DATASET DETAILS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # only show the dataset table if project has selected a dataset
    if session_state.project.datasets:
        df = create_dataframe(session_state.project.datasets, column_names=session_state.project.column_names,
                              sort=True, sort_by='ID', asc=True, date_time_format=True)
        df_loc = df.loc[:, "ID":"Date/Time"]

        styler = df_loc.style

        # >>>> Dataset table placeholders
        dataset_table_col1, dataset_table_col2 = st.columns([3, 0.5])

        dataset_table_col1.write("### **Datasets**")
        dataset_table_col2.write(
            f"### Total datasets: {len(session_state.project.datasets)}")

        st.table(styler.set_properties(**{'text-align': 'center'}).set_table_styles(
            [dict(selector='th', props=[('text-align', 'center')])]))    # >>>>>>>>>>PANDAS DATAFRAME for DATASET DETAILS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    else:
        st.warning("No dataset selected for this project yet! "
                   "Please add one by selecting from the options below.")

    # >>>>>>>>>> Buttons for dataset paginations >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def to_dataset_pages(pagination: IntEnum):
        session_state.project_dashboard_pagination = pagination

    st.markdown("___")
    st.header("Add dataset:")
    st.subheader(
        "Option 1: Create new dataset to add to current project")
    st.button("Create new dataset",
              key='btn_create_new_dataset', on_click=to_dataset_pages,
              args=(ProjectDashboardPagination.CreateNewDataset,))

    st.subheader(
        "Option 2: Upload a labeled dataset for the current project")
    st.button("Upload labeled dataset",
              key='btn_upload_labeled', on_click=to_dataset_pages,
              args=(ProjectDashboardPagination.UploadLabeledDataset,))

    st.subheader(
        "Option 3: Add any of the uploaded existing datasets to the current project")
    st.button("Add existing dataset",
              key="btn_add_existing_dataset", on_click=to_dataset_pages,
              args=(ProjectDashboardPagination.AddExistingDataset,))

    # only show this option if already has project_dataset
    if session_state.project.datasets:
        project_dataset_chosen = list(
            session_state.project.dataset_dict.keys())
        st.subheader("Option 4: Select a project dataset to add more images")
        selected_dataset = st.selectbox("Project datasets", options=project_dataset_chosen,
                                        key='selected_dataset')
        # must set a new session_state to persist to the next page
        session_state.dataset_chosen = selected_dataset
        st.button("Proceed to add images to the selected project dataset",
                  key='btn_proceed_add_images', on_click=to_dataset_pages,
                  args=(ProjectDashboardPagination.AddImageToProjectDataset,))


def index(RELEASE=True):
    # ************************ TEST ************************
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
        project_id_tmp = 30
        logger.debug(f"Entering Project {project_id_tmp}")

        # session_state.append_project_flag = ProjectPermission.ViewOnly

        if "project" not in session_state:
            session_state.project = Project(project_id_tmp)
            logger.debug("Inside")
        if 'user' not in session_state:
            session_state.user = User(1)
    # ************************ TEST ************************

    if 'project_dashboard_pagination' not in session_state:
        session_state.project_dashboard_pagination = ProjectDashboardPagination.ExistingProjectDashboard

    # >>>>>>>>>> Pagination Functions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def add_existing_dataset(**kwargs):
        """Similar to the dataset selection in new_project.py"""
        existing_dataset, dataset_table_column_names = query_dataset_list()
        # to handle situation without any dataset uploaded yet
        if not existing_dataset:
            st.warning("No dataset has been uploaded into the application yet. "
                       "Please go back to relevant pages to create a new dataset first.")
            st.stop()
        dataset_dict = get_dataset_name_list(existing_dataset)
        project_dataset_chosen = list(
            session_state.project.dataset_dict.keys())
        all_datasets = list(dataset_dict.keys())
        for d in project_dataset_chosen:
            # avoid showing selected project datasets to the user to add again
            all_datasets.remove(d)

        st.header("All existing datasets")
        df = create_dataframe(existing_dataset,
                              column_names=dataset_table_column_names,
                              date_time_format=True)
        df = df.loc[:, "ID":"Date/Time"]

        # GET color from active theme
        df_row_highlight_color = get_df_row_highlight_color()

        def highlight_row(x, selections):
            if x.Name in selections:
                return [f'background-color: {df_row_highlight_color}'] * len(x)
            else:
                return ['background-color: '] * len(x)

        styler = df.style.apply(
            highlight_row, selections=project_dataset_chosen, axis=1)

        # >>>>DATAFRAME
        st.dataframe(styler.set_properties(**{'text-align': 'center'}).set_table_styles(
            [dict(selector='th', props=[('text-align', 'center')])]),
            width=1920, height=900)

        datasets_to_add = st.multiselect(
            "Select dataset(s) to add to current project",
            options=all_datasets, key="datasets_to_add",
            help="""Add more dataset to the project. The colored row in the
            table is the project dataset chosen during project creation.""")

        if datasets_to_add:
            st.write("### Dataset chosen:")
            for idx, data in enumerate(datasets_to_add):
                st.write(f"{idx+1}. {data}")
        else:
            st.info("No dataset selected")

        def submit_add_existing_dataset():
            if not datasets_to_add:
                st.warning("No dataset selected to add yet.")
                st.stop()

            session_state.project.insert_project_dataset(
                datasets_to_add, dataset_dict)
            logger.info(f"Inserted project datasets '{datasets_to_add}' "
                        f"for Project {session_state.project.id} into project_dataset table")
            st.success("Successfully added the selected datasets into project.")
            with st.spinner("Refreshing page ..."):
                session_state.project.refresh_project_details()
                session_state.project_dashboard_pagination = ProjectDashboardPagination.ExistingProjectDashboard
            # rerun to show the refreshed details
            st.experimental_rerun()

        if st.button("Submit", key="submit_add_existing_dataset"):
            submit_add_existing_dataset()

    # >>>>>>>>>> Pagination >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    dashboard_pagination2func = {
        ProjectDashboardPagination.ExistingProjectDashboard: dashboard,
        ProjectDashboardPagination.AddExistingDataset: add_existing_dataset,
        ProjectDashboardPagination.CreateNewDataset: new_dataset,
        ProjectDashboardPagination.UploadLabeledDataset: new_dataset,
        ProjectDashboardPagination.AddImageToProjectDataset: new_dataset,
    }

    # >>>>>>>>>> Back to Dashboard button >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def to_dashboard():
        session_state.project_dashboard_pagination = ProjectDashboardPagination.ExistingProjectDashboard

    pagination = session_state.project_dashboard_pagination

    if pagination != ProjectDashboardPagination.ExistingProjectDashboard:
        st.sidebar.button("Back to Project Dashboard", key="to_project_dashboard_sidebar",
                          on_click=to_dashboard)

    # >>>>>>>>>> Enter page >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.debug(f"Enter page: {pagination = }")
    is_new_project = False  # this is for an existing project
    if pagination == ProjectDashboardPagination.AddImageToProjectDataset:
        # is_existing_dataset is required to add image to existing project_dataset,
        #  and the user can choose whether the dataset is labeled or not in the page
        is_existing_dataset = True
    elif pagination == ProjectDashboardPagination.UploadLabeledDataset:
        is_existing_dataset = False
        session_state.is_labeled = True
    else:
        is_existing_dataset = False
        session_state.is_labeled = False
    dashboard_pagination2func[pagination](is_new_project=is_new_project,
                                          is_existing_dataset=is_existing_dataset)

    # st.write(vars(session_state.project))


if __name__ == "__main__":
    # Set to wide page layout for debugging on this page
    # layout = 'wide'
    # st.set_page_config(page_title="Integrated Vision Inspection System",
    #                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)
    if st._is_running_with_streamlit:
        # This is set to False for debugging purposes
        # when running Streamlit directly from this page
        index(RELEASE=False)
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
