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

import sys
from pathlib import Path
from enum import IntEnum
from time import sleep
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
from core.utils.code_generator import get_random_string
from core.utils.log import logger  # logger
from core.utils.helper import create_dataframe, get_df_row_highlight_color, get_textColor, current_page, non_current_page
from core.utils.form_manager import remove_newline_trailing_whitespace
from data_manager.database_manager import init_connection
from data_manager.annotation_type_select import annotation_sel
from data_manager.dataset_management import NewDataset, query_dataset_list, get_dataset_name_list
from project.project_management import NewProject, ProjectPagination, NewProjectPagination, new_project_nav
from data_editor.editor_management import Editor, NewEditor
from data_editor.editor_config import editor_config
from pages.sub_pages.dataset_page.new_dataset import new_dataset

# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>


# >>>> Variable Declaration >>>>
new_project = {}  # store
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

    # ******** SESSION STATE *********************************************************
    if session_state.new_project.has_submitted:

        def to_editor_config():
            session_state.new_project_pagination = NewProjectPagination.EditorConfig

        st.button(
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
        f"### __Project ID:__ {session_state.new_project.id}")

    infocol1.write("## __Project Information :__")

    # >>>> CHECK IF NAME EXISTS CALLBACK >>>>
    def check_if_name_exist(field_placeholder, conn):
        context = {'column_name': 'name',
                   'value': session_state.new_project_name}
        logger.debug(context)
        if session_state.new_project_name:
            if session_state.new_project.check_if_exists(context, conn):
                session_state.new_project.name = None
                field_placeholder['new_project_name'].error(
                    f"Project name used. Please enter a new name")
                sleep(1)
                field_placeholder['new_project_name'].empty()
                logger.error(f"Project name used. Please enter a new name")
            else:
                session_state.new_project.name = session_state.new_project_name
                logger.info(f"Project name fresh and ready to rumble")
        else:
            pass

    with infocol2:

        # **** PROJECT TITLE****
        st.text_input(
            "Project Title", key="new_project_name",
            help="Enter the name of the project",
            on_change=check_if_name_exist, args=(place, conn,))
        place["new_project_name"] = st.empty()

        # **** PROJECT DESCRIPTION (OPTIONAL) ****
        description = st.text_area(
            "Description (Optional)", key="new_project_desc",
            help="Enter the description of the project")

        if description:
            session_state.new_project.desc = remove_newline_trailing_whitespace(
                description)
        else:
            pass

        # **** DEPLOYMENT TYPE and EDITOR BASE TEMPLATE LOAD ****
        v = annotation_sel()

        if None not in v:
            (deployment_type, editor_base_config) = v

            session_state.new_editor.editor_config = editor_base_config['config']

            # TODO Remove deployment id
            session_state.new_project.deployment_type = deployment_type
            session_state.new_project.query_deployment_id()

        else:
            session_state.new_editor.editor_config = None
            session_state.new_project.deployment_type = None
            session_state.new_project.deployment_id = None

        place["new_project_deployment_type"] = st.empty()

    # <<<<<<<< New Project INFO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>>>>>> Choose Dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    datasetcol1.write("## __Dataset :__")

    # ******************************* Right Column to select dataset *******************************

    with datasetcol3:
        session_state.new_project.dataset_chosen = st.multiselect(
            "Dataset List", key="new_project_dataset_chosen",
            options=dataset_dict, help="Assign dataset to the project")
        place["new_project_dataset_chosen"] = st.empty()

    # >>>> CREATE NEW DATASET AND SELECT DATASET >>>>>>>>>>>>>>>>>>>>>>>>>>>>

        def to_new_dataset_page():
            session_state.new_project_pagination = NewProjectPagination.NewDataset

        st.button("Create New Dataset", key="create_new_dataset_from_new_project",
                  on_click=to_new_dataset_page, help="Create new dataset")

        # >>>> DISPLAY CHOSEN DATASET>>>>
        st.write("### Dataset chosen:")
        if len(session_state.new_project.dataset_chosen) > 0:
            for idx, data in enumerate(session_state.new_project.dataset_chosen):
                st.write(f"{idx+1}. {data}")

        elif len(session_state.new_project.dataset_chosen) == 0:
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

    # ******************* Left Column to show full list of dataset and selection *******************

    # **************************************** DATASET PAGINATION ****************************************

    # >>>> PAGINATION CALLBACK >>>>
    def next_page():
        session_state.new_project_dataset_page += 1

    def prev_page():
        session_state.new_project_dataset_page -= 1

    num_dataset_per_page = 10
    num_dataset_page = len(
        dataset_dict) // num_dataset_per_page

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
        'new_project_name': session_state.new_project.name,
        'new_project_deployment_type': session_state.new_project.deployment_type,
        'new_project_dataset_chosen': session_state.new_project.dataset_chosen
    }

    submit_col1, upload_col, submit_col2 = st.columns([2.5, 0.5, 0.5])

    def new_project_submit(dataset_dict, labeled=False):
        # if this is True, we will send to New Dataset page for uploading
        session_state.is_labeled = labeled

        if labeled:
            # if uploading labeled dataset, then no need to check the dataset_chosen field
            del context['new_project_dataset_chosen']
        session_state.new_project.has_submitted = session_state.new_project.check_if_field_empty(
            context, field_placeholder=place, name_key='new_project_name')

        if session_state.new_project.has_submitted:
            # TODO #13 Load Task into DB after creation of project
            # NOTE: dataset_dict is None when user choose to upload labeled dataset,
            #  this is to skip inserting project dataset that has not been chosen
            if session_state.new_project.initialise_project(dataset_dict):
                # Updated with Actual Project ID from DB
                session_state.new_editor.project_id = session_state.new_project.id
                # deployment type now IntEnum
                if session_state.new_editor.init_editor(session_state.new_project.deployment_type):
                    session_state.new_project.editor = Editor(
                        session_state.new_project.id, session_state.new_project.deployment_type)
                    success_place.success(
                        f"Successfully stored **{session_state.new_project.name}** project information in database")
                    sleep(1)
                    success_place.empty()
                    if labeled:
                        # send to the NewDataset page to upload labeled dataset
                        session_state.new_project_pagination = NewProjectPagination.NewDataset
                    else:
                        # if not uploading new labeled dataset then go to Editor
                        session_state.new_project_pagination = NewProjectPagination.EditorConfig
                else:
                    success_place.error(
                        f"Failed to stored **{session_state.new_editor.name}** editor config in database")

            else:
                success_place.error(
                    f"Failed to stored **{session_state.new_project.name}** project information in database")
    # TODO #72 Change to 'Update' when 'has_submitted' == True
    submit_button = submit_col2.button(
        "Submit", key="submit", on_click=new_project_submit,
        kwargs={'dataset_dict': dataset_dict, 'labeled': False})

    with datasetcol3:
        with st.expander("NOTES about uploading a labeled dataset"):
            st.info("""If you choose to upload a labeled dataset, you must first fill
            up the project title and select a template for the deployment type of the
            computer vision task.""")
    with upload_col:
        st.button("Upload Labeled Dataset", key='btn_upload_labeled_data',
                  on_click=new_project_submit,
                  kwargs={'dataset_dict': None, 'labeled': True})

    # >>>> Removed
    # session_state.new_project.has_submitted = False

    col1, col2 = st.columns(2)
    col1.write(vars(session_state.new_project))
    # col2.write(vars(session_state.new_editor))
    col2.write(context)
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

    new_project_home_col1, new_project_home_col2 = st.columns([3, 0.5])
    logger.debug(
        f" New Project Pagination: {session_state.new_project_pagination}")
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

        new_project_home_col2.button(
            "Done", key='back_to_project_dashboard', on_click=to_project_dashboard)
        logger.debug(
            f"Project ID before editor config: {session_state.new_project.id},{session_state.new_project.editor}")
        new_project_page[session_state.new_project_pagination](
            session_state.new_project)
    else:
        new_project_page[session_state.new_project_pagination]()

    new_project_back_button_place = st.empty()

    # >>>> RETURN TO ENTRY PAGE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if session_state.new_project_pagination != NewProjectPagination.Entry:

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
    layout = 'wide'
    st.set_page_config(page_title="Integrated Vision Inspection System",
                       page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

    if st._is_running_with_streamlit:
        # initialise connection to Database
        conn = init_connection(**st.secrets["postgres"])
        index(RELEASE=False, conn=conn)
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
