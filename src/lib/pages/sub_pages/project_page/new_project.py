"""
Title: New Project Page
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

import gc
import math
import sys
from pathlib import Path
from enum import IntEnum
from time import sleep
from typing import Any, Dict, List
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state as session_state

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
# TEST_MODULE_PATH = SRC / "test" / "test_page" / "module"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

from path_desc import chdir_root
from annotation.annotation_management import Annotations, NewTask, Task
from core.utils.code_generator import get_random_string
from core.utils.log import logger  # logger
from core.utils.helper import create_dataframe, get_df_row_highlight_color, get_textColor, current_page, non_current_page
from core.utils.form_manager import remove_newline_trailing_whitespace
from data_manager.database_manager import init_connection
from data_manager.annotation_type_select import annotation_sel
from data_manager.dataset_management import Dataset, NewDataset, query_dataset_list, get_dataset_name_list
from project.project_management import NewProject, Project, ProjectPagination, NewProjectPagination, new_project_nav, query_project_dataset_annotations, query_project_dataset_task_count, query_project_dataset_tasks
from data_editor.editor_management import Editor, NewEditor
from data_editor.editor_config import editor_config
from pages.sub_pages.dataset_page.new_dataset import new_dataset

# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>


# >>>> Variable Declaration >>>>
place = {}
DEPLOYMENT_TYPE = ("", "Image Classification", "Object Detection with Bounding Boxes",
                   "Semantic Segmentation with Polygons")


chdir_root()  # change to root directory


def new_project_entry_page(conn=None):
    if not conn:
        conn = init_connection(**st.secrets["postgres"])

    # >>>> INIT >>>>
    # ********* QUERY DATASET ********************************************
    existing_dataset, dataset_table_column_names = query_dataset_list()
    dataset_dict = get_dataset_name_list(existing_dataset)
    # ********* QUERY DATASET ********************************************

    # ******** SESSION STATE ***********************************************************

    if "new_project" not in session_state:
        session_state.new_project = NewProject(get_random_string(length=8))

    if 'new_editor' not in session_state:
        session_state.new_editor = NewEditor(get_random_string(length=8))
        # set random project ID before getting actual from Database

    if 'project_status' not in session_state:
        session_state.project_status = ProjectPagination.New
    else:
        session_state.project_status = ProjectPagination.New

    if 'is_labeled' not in session_state:
        # to check whether user choose to upload a labeled dataset
        session_state.is_labeled = False

    new_project: NewProject = session_state.new_project
    new_editor: NewEditor = session_state.new_editor

    # ******** SESSION STATE *********************************************************
    if new_project.has_submitted:

        def to_editor_config():
            session_state.new_project_pagination = NewProjectPagination.EditorConfig

        st.sidebar.button(
            "Next", key="new_project_to_editor", on_click=to_editor_config)
    # Page title
    st.write("# __Add New Project__")
    st.markdown("___")

    # ************COLUMN PLACEHOLDERS *****************************************************
    # right-align the project ID relative to the page
    id_blank, id_right = st.columns([3, 1])

    # Columns for Project Information
    infocol1, infocol2, infocol3 = st.columns([1.5, 3.5, 0.5])

    # include options to create new dataset on this page
    # create 2 columns for "New Data Button"
    datasetcol1, datasetcol2, datasetcol3, _ = st.columns(
        [1.5, 1.75, 1.75, 0.5])

    # COLUMNS for Dataframe buttons
    _, button_col1, _, button_col2, _, button_col3, _ = st.columns(
        [1.5, 0.15, 0.5, 0.45, 0.5, 0.15, 2.25])

    # ************COLUMN PLACEHOLDERS *****************************************************
    # <<<< INIT <<<<

# >>>> New Project INFO >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    id_right.write(
        f"### __Project ID:__ {new_project.id}")

    infocol1.write("## __Project Information :__")

    # >>>> CHECK IF NAME EXISTS CALLBACK >>>>
    def check_if_name_exist(field_placeholder, conn):
        context = {'column_name': 'name',
                   'value': session_state.new_project_name}
        logger.debug(context)
        if session_state.new_project_name:
            if new_project.check_if_exists(context, conn):
                new_project.name = None
                field_placeholder['new_project_name'].error(
                    f"Project name used. Please enter a new name")
                sleep(1)
                field_placeholder['new_project_name'].empty()
                logger.error(f"Project name used. Please enter a new name")
            else:
                new_project.name = session_state.new_project_name
                logger.info(f"Project name fresh and ready to rumble")
        else:
            pass

    with infocol2:

        # **** PROJECT TITLE****
        st.text_input(
            "Project Title", max_chars=21, key="new_project_name",
            help="Enter the name of the project. Name should be less than 21 characters long",
            on_change=check_if_name_exist, args=(place, conn,))
        place["new_project_name"] = st.empty()

        # **** PROJECT DESCRIPTION (OPTIONAL) ****
        description = st.text_area(
            "Description (Optional)", key="new_project_desc",
            help="Enter the description of the project")

        if description:
            new_project.desc = remove_newline_trailing_whitespace(
                description)

        # **** DEPLOYMENT TYPE and EDITOR BASE TEMPLATE LOAD ****
        v = annotation_sel()

        if None not in v:
            (deployment_type, editor_base_config) = v

            new_editor.editor_config = editor_base_config['config']

            # TODO Remove deployment id
            new_project.deployment_type = deployment_type
            new_project.query_deployment_id()

        else:
            new_editor.editor_config = None
            new_project.deployment_type = None
            new_project.deployment_id = None

        place["new_project_deployment_type"] = st.empty()

    # <<<<<<<< New Project INFO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>>>>>> Choose Dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    datasetcol1.write("## __Dataset :__")

    # ******************************* Right Column to select dataset *******************************

    with datasetcol3:
        dataset_chosen = st.multiselect(
            "Dataset List", key="new_project_dataset_chosen",
            options=dataset_dict, help="Assign dataset to the project")
        place["new_project_dataset_chosen"] = st.empty()

    # >>>> CREATE NEW DATASET AND SELECT DATASET >>>>>>>>>>>>>>>>>>>>>>>>>>>>

        def to_new_dataset_page():
            session_state.new_project_pagination = NewProjectPagination.NewDataset

        st.button("Create New Dataset", key="create_new_dataset_from_new_project",
                  on_click=to_new_dataset_page, help="Create new dataset")

        upload_place = st.empty()

        # >>>> DISPLAY CHOSEN DATASET>>>>
        st.write("### Dataset chosen:")
        if not new_project.deployment_type:
            st.info("""You may select a deployment type first to be able to 
            use any existing labeled data if available.""")

        if dataset_chosen:
            annotated_dataset_id2_project_id: Dict[int, int] = {}
            for idx, data in enumerate(dataset_chosen):
                st.write(f"{idx+1}. {data}")

                # check related projects only if the user has chosen a deployment type
                if new_project.deployment_type:
                    dataset_id_chosen = dataset_dict[data].ID
                    related_projects = Dataset.query_related_projects(
                        dataset_id_chosen, new_project.deployment_type,
                        is_labelled=True
                    )
                else:
                    continue

                if st.checkbox("Use existing labeled data?",
                               key=f'use_existing_annotations_{idx}'):
                    if not related_projects:
                        st.info("There is no related project for this dataset "
                                "with the same deployment type.")
                        logger.info(f"There is no related projects for dataset {data} "
                                    f"of ID {dataset_id_chosen} with the same deployment type")
                        continue

                    project_names = []
                    project_ids = []
                    display_names = []
                    for r in related_projects:
                        name = r.name
                        project_id = r.id

                        labeled_counts = query_project_dataset_task_count(
                            project_id, dataset_id_chosen, is_labelled=True)
                        if not labeled_counts:
                            st.info(
                                "There is no existing labeled data for this dataset.")
                            logger.info(f"Project ID {project_id} has no labeled data "
                                        f"for the dataset: {data}")
                            continue

                        display_names.append(f"{name} ({labeled_counts})")
                        project_names.append(name)
                        project_ids.append(project_id)

                    if not project_ids:
                        st.info(
                            "There is no existing annotations for this dataset")
                        continue

                    selected_display_name = st.radio(
                        "Select a project", options=display_names,
                        key=f'selected_labeled_project_{idx}',
                        help="The number in the brackets is the number of labeled images")
                    selected_idx = display_names.index(
                        selected_display_name)
                    selected_project_id = project_ids[selected_idx]
                    logger.info(f"Selected Project '{selected_display_name}' "
                                f"of ID: {selected_project_id} for dataset '{data}'")
                    # dataset_info = {
                    #     'dataset_id': dataset_id_chosen,
                    #     'project_id': selected_project_id}
                    annotated_dataset_id2_project_id[dataset_id_chosen] = selected_project_id
        else:
            st.info("No dataset selected")

    # ******************************* Right Column to select dataset *******************************

    # ******************* Left Column to show full list of dataset and selection *******************

    if "new_project_dataset_page" not in session_state:
        session_state.new_project_dataset_page = 0

    # only show the DataFrame of datasets if there is already existing dataset created
    if existing_dataset:
        with datasetcol2:
            start = 10 * session_state.new_project_dataset_page
            end = start + 10

            # >>>>>>>>>>PANDAS DATAFRAME >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            df = create_dataframe(existing_dataset,
                                  column_names=dataset_table_column_names,
                                  date_time_format=True)

            df_loc = df.loc[:, "ID":"Date/Time"]
            df_slice = df_loc.iloc[start:end]

            # GET color from active theme
            df_row_highlight_color = get_df_row_highlight_color()

            def highlight_row(x, selections):

                if x.Name in selections:

                    return [f'background-color: {df_row_highlight_color}'] * len(x)
                else:
                    return ['background-color: '] * len(x)

            styler = df_slice.style.apply(
                highlight_row, selections=session_state.new_project_dataset_chosen, axis=1)

            # >>>>DATAFRAME
            st.table(styler.set_properties(**{'text-align': 'center'}).set_table_styles(
                [dict(selector='th', props=[('text-align', 'center')])]))
            # st.dataframe(styler, height=600)

    # ******************* Left Column to show full list of dataset and selection *******************

    # **************************************** DATASET PAGINATION ****************************************

    # >>>> PAGINATION CALLBACK >>>>
    def next_page():
        session_state.new_project_dataset_page += 1

    def prev_page():
        session_state.new_project_dataset_page -= 1

    num_dataset_per_page = 10
    num_dataset_page = math.ceil(len(
        dataset_dict) / num_dataset_per_page)

    print(f"{num_dataset_page = }")
    print(f"{session_state.new_project_dataset_page = }")

    if num_dataset_page > 1:
        if session_state.new_project_dataset_page < num_dataset_page:
            button_col3.button(">", on_click=next_page)
        else:
            # this makes the empty column show up on mobile
            button_col3.write("")

        if session_state.new_project_dataset_page > 0:
            button_col1.button("<", on_click=prev_page)
        else:
            # this makes the empty column show up on mobile
            button_col1.write("")

        button_col2.write(
            f"Page {1+session_state.new_project_dataset_page} of {num_dataset_page}")
    # **************************************** DATASET PAGINATION ****************************************

    # ******************************** SUBMISSION *************************************************
    success_place = st.empty()
    context = {
        'new_project_name': new_project.name,
        'new_project_deployment_type': new_project.deployment_type,
        'new_project_dataset_chosen': dataset_chosen
    }

    submit_col1, _, submit_col2 = st.columns([2.5, 0.5, 0.5])

    def new_project_submit(dataset_chosen=dataset_chosen, dataset_dict=dataset_dict, labeled=False):
        # if this is True, we will send to New Dataset page for uploading
        if labeled:
            # set this to True to tell new_dataset page about uploading labeled data
            session_state.is_labeled = labeled
            # if uploading labeled dataset, then no need to check the dataset_chosen field
            del context['new_project_dataset_chosen']
        new_project.has_submitted = new_project.check_if_field_empty(
            context, field_placeholder=place, name_key='new_project_name')

        if new_project.has_submitted:
            # TODO #13 Load Task into DB after creation of project
            # NOTE: dataset_dict is None when user choose to upload labeled dataset,
            #  this is to skip inserting project dataset that has not been chosen
            if labeled:
                dataset_dict = None
                dataset_chosen = None
            if new_project.initialise_project(
                    dataset_chosen, dataset_dict, annotated_dataset_id2_project_id):
                # Updated with Actual Project ID from DB
                new_editor.project_id = new_project.id
                # deployment type now IntEnum
                if new_editor.init_editor(new_project.deployment_type):
                    new_project.editor = Editor(
                        new_project.id, new_project.deployment_type)
                    success_place.success(
                        f"Successfully stored **{new_project.name}** project information in database")
                    sleep(1)
                    success_place.empty()
                    if labeled:
                        # send to the NewDataset page to upload labeled dataset
                        session_state.new_project_pagination = NewProjectPagination.NewDataset
                    else:
                        if annotated_dataset_id2_project_id:
                            # No need to insert project_dataset or update the tasks
                            # as they are already done in initialise_project()

                            # initialize Project and remove NewProject from session_states
                            new_project_id = new_project.id
                            NewProject.reset_new_project_page()
                            session_state.project = Project(new_project_id)

                            # update the current Project's EditorConfig based on
                            # the inserted annotations
                            logger.info(
                                "Updating EditorConfig based on the annotations")
                            session_state.project.update_editor_config(
                                refresh_project=True)

                            # enter the project overview page directly
                            logger.info(
                                f"Entering Project {session_state.project.id}")
                            session_state.project_pagination = ProjectPagination.Existing
                        else:
                            # if not uploading new labeled dataset then go to Editor
                            session_state.new_project_pagination = NewProjectPagination.EditorConfig
                    gc.collect()
                    st.experimental_rerun()
                else:
                    success_place.error(
                        f"Failed to stored **{new_editor.name}** editor config in database")

            else:
                success_place.error(
                    f"Failed to stored **{new_project.name}** project information in database")
        else:
            st.stop()

    with upload_place.container():
        if st.button("Upload Labeled Dataset", key='btn_upload_labeled_data'):
            new_project_submit(labeled=True)
        with st.expander("NOTES about uploading a labeled dataset"):
            st.info("""If you choose to upload a labeled dataset, you must first fill
            up the project title and select a template for the deployment type of the
            computer vision task.""")

    # put button at the bottom to allow other things to render first
    # in case error happens and st.stop() earlier
    # TODO #72 Change to 'Update' when 'has_submitted' == True
    if submit_col2.button("Submit", key="submit"):
        new_project_submit()

    # >>>> Removed
    # session_state.new_project.has_submitted = False

    # col1, col2 = st.columns(2)
    # col1.write(vars(session_state.new_project))
    # col2.write(vars(session_state.new_editor))
    # col2.write(context)
    # col2.write(dataset_dict)


def index(RELEASE=True, conn=None):
    if not conn:
        conn = init_connection(**st.secrets["postgres"])

    new_project_page = {
        NewProjectPagination.Entry: new_project_entry_page,
        NewProjectPagination.EditorConfig: editor_config,
        NewProjectPagination.NewDataset: new_dataset
    }

    if 'new_project_pagination' not in session_state:
        session_state.new_project_pagination = NewProjectPagination.Entry

    # new_project_home_col1, new_project_home_col2 = st.columns([3, 0.5])
    editor_config_submit_place = st.sidebar.empty()
    logger.debug(
        f"Navigator: {session_state.new_project_pagination = }")
    # ******************** TOP PAGE NAV *******************************************************************************************************
    if (session_state.new_project_pagination == NewProjectPagination.Entry) or (session_state.new_project_pagination == NewProjectPagination.NewDataset):
        color = [current_page, non_current_page]
    else:
        color = [non_current_page, current_page]
    try:
        textColor = get_textColor()
        logger.debug(f"{color},{textColor}")
        new_project_nav(color, textColor)
    except:
        pass

    # ******************** TOP PAGE NAV *******************************************************************************************************

    # NOTE whether to allow going back to Home Page for all pages ???????
    if session_state.new_project_pagination == NewProjectPagination.EditorConfig:

        def to_project_dashboard():
            """Callback to return back to the Project Dashboard
            """
            NewProject.reset_new_project_page()
            session_state.project_pagination = ProjectPagination.Dashboard
            session_state.new_project_pagination = NewProjectPagination.Entry

        editor_config_submit_place.button(
            "Back to Project Dashboard", key='back_to_project_dashboard',
            on_click=to_project_dashboard)
        logger.debug(
            f"Project ID before editor config: {session_state.new_project.id},{session_state.new_project.editor}")
        new_project_page[session_state.new_project_pagination](
            session_state.new_project)
    else:
        new_project_page[session_state.new_project_pagination]()

    new_project_back_button_place = st.sidebar.empty()

    # >>>> RETURN TO ENTRY PAGE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # NOTE: not showing this "Back" button for uploading labeled dataset because
    # the project has already been submitted there to associate with the uploaded labeled dataset
    # and it's not necessary to come back to new_project page here
    if session_state.new_project_pagination == NewProjectPagination.NewDataset \
            and not session_state.is_labeled:

        def to_new_project_entry_page():
            NewDataset.reset_new_dataset_page()
            session_state.new_project_pagination = NewProjectPagination.Entry

        new_project_back_button_place.button("Back", key="back_to_entry_page",
                                             on_click=to_new_project_entry_page)

    else:
        new_project_back_button_place.empty()

    # st.write(session_state.new_project_pagination)


if __name__ == "__main__":
    # DEFINE wide page layout for debugging on this page directly
    # layout = 'wide'
    # st.set_page_config(page_title="Integrated Vision Inspection System",
    #                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

    if st._is_running_with_streamlit:
        # initialise connection to Database
        conn = init_connection(**st.secrets["postgres"])
        index(RELEASE=False, conn=conn)
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
