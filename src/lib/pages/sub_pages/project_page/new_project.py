"""
Title: New Project Page
Date: 5/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
from enum import IntEnum
from time import sleep
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state as session_state
from core.webcam.webcam_webrtc import show
from data_editor.editor_config import editor_config
from pages.sub_pages.dataset_page.new_dataset import new_dataset


# DEFINE Web APP page configuration
# layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

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
from core.utils.log import log_info, log_error  # logger
from core.utils.helper import create_dataframe, get_df_row_highlight_color
from data_manager.database_manager import init_connection
from data_manager.annotation_type_select import annotation_sel
from data_manager.dataset_management import NewDataset, query_dataset_list, get_dataset_name_list
from project.project_management import NewProject, ProjectPagination, NewProjectPagination
from data_editor.editor_management import Editor, NewEditor

# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


# >>>> Variable Declaration >>>>
new_project = {}  # store
place = {}
DEPLOYMENT_TYPE = ("", "Image Classification", "Object Detection with Bounding Boxes",
                   "Semantic Segmentation with Polygons", "Semantic Segmentation with Masks")


chdir_root()  # change to root directory


def new_project_entry_page():

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
    if 'project_pagination' not in session_state:
        session_state.project_pagination = ProjectPagination.New
    else:
        session_state.project_pagination = ProjectPagination.New
    # ******** SESSION STATE *********************************************************

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
        context = {'column_name': 'name', 'value': session_state.name}

        if session_state.name:
            if session_state.new_project.check_if_exists(context, conn):
                session_state.new_project.name = None
                field_placeholder['name'].error(
                    f"Project name used. Please enter a new name")
                sleep(1)
                field_placeholder['name'].empty()
                log_error(f"Project name used. Please enter a new name")
            else:
                session_state.new_project.name = session_state.name
                log_info(f"Project name fresh and ready to rumble")
        else:
            pass

    with infocol2:

        # **** PROJECT TITLE****
        st.text_input(
            "Project Title", key="name", help="Enter the name of the project", on_change=check_if_name_exist, args=(place, conn,))
        place["name"] = st.empty()

        # **** PROJECT DESCRIPTION (OPTIONAL) ****
        description = st.text_area(
            "Description (Optional)", key="desc", help="Enter the description of the project")
        if description:
            session_state.new_project.desc = description
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

        place["deployment_type"] = st.empty()

    # <<<<<<<< New Project INFO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>>>>>> Choose Dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    datasetcol1.write("## __Dataset :__")

    # ******************************* Right Column to select dataset *******************************

    with datasetcol3:

        session_state.new_project.dataset_chosen = st.multiselect(
            "Dataset List", key="dataset_chosen", options=dataset_dict, help="Assign dataset to the project")
        place["dataset_chosen"] = st.empty()

        # TODO #55 Button to create new dataset

    # >>>> CREATE NEW DATASET AND SELECT DATASET >>>>>>>>>>>>>>>>>>>>>>>>>>>>
        def to_new_dataset_page():
            session_state.new_project_pagination = NewProjectPagination.NewDataset

        st.button("Create New Dataset", key="create_new_dataset_from_new_project",
                  on_click=to_new_dataset_page, help="Create new dataset")

        # >>>> DISPLAY CHOSEN DATASET>>>>
        st.write("### Dataset choosen:")
        if len(session_state.new_project.dataset_chosen) > 0:
            for idx, data in enumerate(session_state.new_project.dataset_chosen):
                st.write(f"{idx+1}. {data}")

        elif len(session_state.new_project.dataset_chosen) == 0:
            st.info("No dataset selected")

    # ******************************* Right Column to select dataset *******************************

    # ******************* Left Column to show full list of dataset and selection *******************

    if "dataset_page" not in session_state:
        session_state.dataset_page = 0

    with datasetcol2:
        start = 10 * session_state.dataset_page
        end = start + 10

        # >>>>>>>>>>PANDAS DATAFRAME >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        df = create_dataframe(existing_dataset,
                              column_names=dataset_table_column_names, date_time_format=True)

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
            highlight_row, selections=session_state.dataset_chosen, axis=1)

        # >>>>DATAFRAME
        st.table(styler.set_properties(**{'text-align': 'center'}).set_table_styles(
            [dict(selector='th', props=[('text-align', 'center')])]))

    # ******************* Left Column to show full list of dataset and selection *******************

    # **************************************** DATASET PAGINATION ****************************************

    # >>>> PAGINATION CALLBACK >>>>
    def next_page():
        session_state.dataset_page += 1

    def prev_page():
        session_state.dataset_page -= 1

    num_dataset_per_page = 10
    num_dataset_page = len(
        dataset_dict) // num_dataset_per_page

    if num_dataset_page > 1:
        if session_state.dataset_page < num_dataset_page:
            button_col3.button(">", on_click=next_page)
        else:
            # this makes the empty column show up on mobile
            button_col3.write("")

        if session_state.dataset_page > 0:
            button_col1.button("<", on_click=prev_page)
        else:
            # this makes the empty column show up on mobile
            button_col1.write("")

        button_col2.write(
            f"Page {1+session_state.dataset_page} of {num_dataset_page}")
    # **************************************** DATASET PAGINATION ****************************************

    # TODO #52 Add Editor Config

    # ******************************** SUBMISSION *************************************************
    success_place = st.empty()
    context = {'name': session_state.new_project.name,
               'deployment_type': session_state.new_project.deployment_type, 'dataset_chosen': session_state.new_project.dataset_chosen}

    submit_col1, submit_col2 = st.columns([3, 0.5])
    submit_button = submit_col2.button("Submit", key="submit")

    # >>>> Removed
    # session_state.new_project.has_submitted = False
    if submit_button:
        session_state.new_project.has_submitted = session_state.new_project.check_if_field_empty(
            context, field_placeholder=place)

        if session_state.new_project.has_submitted:
            # TODO #13 Load Task into DB after creation of project
            if session_state.new_project.initialise_project(dataset_dict):
                session_state.new_editor.project_id = session_state.new_project.id
                if session_state.new_editor.init_editor():
                    session_state.new_project.editor = Editor(
                        session_state.new_project.id, session_state.new_project.deployment_type)
                    success_place.success(
                        f"Successfully stored **{session_state.new_project.name}** project information in database")
                    # session_state.new_project_pagination == NewProjectPagination.EditorConfig #TODO
                else:
                    success_place.error(
                        f"Failed to stored **{session_state.new_editor.name}** editor config in database")

            else:
                success_place.error(
                    f"Failed to stored **{session_state.new_project.name}** project information in database")

    col1, col2 = st.columns(2)
    # col1.write(vars(session_state.new_project))
    # col2.write(vars(session_state.new_editor))
    col2.write(context)
    # col2.write(dataset_dict)


def index():
    new_project_page = {
        NewProjectPagination.Entry: new_project_entry_page,
        NewProjectPagination.NewDataset: new_dataset,
        NewProjectPagination.EditorConfig: editor_config
    }
    if 'new_project_pagination' not in session_state:
        session_state.new_project_pagination = NewProjectPagination.Entry

    if session_state.new_project_pagination == NewProjectPagination.EditorConfig:
        new_project_page[session_state.new_project_pagination](
            session_state.new_project)
    else:
        new_project_page[session_state.new_project_pagination]()

    back_to_entry_page_button_place = st.empty()

    if session_state.new_project_pagination == NewProjectPagination.NewDataset:

        def to_new_project_entry_page():
            NewDataset.reset_new_dataset_page()
            session_state.new_project_pagination = NewProjectPagination.Entry
        back_to_entry_page_button_place.button("Back", key="back_to_entry_page",
                                               on_click=to_new_project_entry_page)

    else:
        back_to_entry_page_button_place.empty()


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        index()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
