"""
Title: Labelling Dashboard
Date: 19/8/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
from enum import IntEnum
from typing import List
from time import sleep
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state as session_state

# DEFINE Web APP page configuration
layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

from data_table import data_table
# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[5]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
from core.utils.helper import create_dataframe, get_df_row_highlight_color, get_textColor, current_page, non_current_page
from core.utils.form_manager import remove_newline_trailing_whitespace
from data_manager.database_manager import init_connection
from data_manager.dataset_management import NewDataset, query_dataset_list, get_dataset_name_list
from annotation.annotation_management import LabellingPagination
from project.project_management import ExistingProjectPagination, ProjectPermission, Project
from data_editor.editor_management import Editor
from data_editor.editor_config import editor_config
from pages.sub_pages.dataset_page.new_dataset import new_dataset

# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


# >>>> Variable Declaration >>>>
new_project = {}  # store
place = {}

chdir_root()  # change to root directory

# **************** DATA TABLE COLUMN CONFIG *********************************************************
all_task_columns = [
    {
        'field': "id",
        'headerName': "ID",
        'headerAlign': "center",
        'align': "center",
        'flex': 70,
        'hideSortIcons': True,

    },
    {
        'field': "Task Name",
        'headerAlign': "center",
        'align': "center",
        'flex': 200,
        'hideSortIcons': False,
    },
    {
        'field': "Created By",
        'headerAlign': "center",
        'align': "center",
        'flex': 130,
        'hideSortIcons': True,
    },
    {
        'field': "Dataset Name",
        'headerAlign': "center",
        'align': "center",
        'flex': 150,
        'hideSortIcons': False,
    },
    {
        'field': "Is Labelled",
        'headerAlign': "center",
        'align': "center",
        'flex': 100,
        'hideSortIcons': True,
        'type': 'boolean',
    },
    {
        'field': "Skipped",
        'headerAlign': "center",
        'align': "center",
        'flex': 100,
        'hideSortIcons': True,
        'type': 'boolean',
    },
    {
        'field': "Date/Time",
        'headerAlign': "center",
        'align': "center",
        'flex': 100,
        'hideSortIcons': False,
        'type': 'date',
    },

]


# **************** DATA TABLE COLUMN CONFIG *********************************************************


def no_labelled(all_task):
    task_queue_dict = Project.get_labelled_task(all_task, False)

    # >>>> TASK QUEUE TABLE >>>>>>>>>>>>>>>>>>>>>>
    # >>>> All task table placeholders
    task_queue_table_col1, task_queue_table_col2 = st.columns([3, 0.5])
    length_of_queue = len(task_queue_dict)
    total_task_length = len(all_task)

    task_queue_table_col1.write(f"### Task Queue")
    task_queue_table_col2.write(
        f"### Total data: {length_of_queue}/{total_task_length}")
    data_table(task_queue_dict, all_task_columns,
               checkbox=False, key='labelled_task_table_key')


def labelled_table(all_task):
    labelled_task_dict = Project.get_labelled_task(all_task, True)

    # >>>> LABELLED TASK TABLE >>>>>>>>>>>>>>>>>>>>>>

    # >>>> All task table placeholders
    labelled_task_table_col1, labelled_task_table_col2 = st.columns([3, 0.5])

    length_of_labelled = len(labelled_task_dict)
    length_of_remaining = len(all_task) - len(labelled_task_dict)

    labelled_task_table_col1.write(f"### Labelled Task")
    labelled_task_table_col2.write(
        f"### Total data: {length_of_labelled}/{length_of_remaining}")
    data_table(labelled_task_dict, all_task_columns,
               checkbox=False, key='labelled_task_table_key')


def all_task_table(all_task):

    # TODO 80 Add Labelling section

    # Query all task +user full name(concat)+ dataset name + annotation status

    # >>>> ALL TASK TABLE >>>>>>>>>>>>>>>>>>>>>>

    # >>>> All task table placeholders
    all_task_table_col1, all_task_table_col2 = st.columns([3, 0.5])

    all_task_table_col1.write(f"### All Task")
    all_task_table_col2.write(f"### Total data: {len(all_task)}")

    data_table(all_task, all_task_columns,
               checkbox=False, key='all_task_table_key')


def index():

    RELEASE = False

    # ****************** TEST ******************************
    if not RELEASE:
        log_info("At Labelling INDEX")

        # ************************TO REMOVE************************
        with st.sidebar.container():
            st.image("resources/MSF-logo.gif", use_column_width=True)
            st.title("Integrated Vision Inspection System", anchor='title')
            st.header(
                "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
            st.markdown("""___""")

        # ************************TO REMOVE************************
        project_id_tmp = 43
        log_info(f"Entering Project {project_id_tmp}")

        # session_state.append_project_flag = ProjectPermission.ViewOnly

        if "project" not in session_state:
            session_state.project = Project(project_id_tmp)
            log_info("Inside")

        # ****************************** HEADER **********************************************
        st.write(f"# {session_state.project.name}")

        project_description = session_state.project.desc if session_state.project.desc is not None else " "
        st.write(f"{project_description}")

        st.markdown("""___""")
        # ****************************** HEADER **********************************************

    st.write(f"## **Labelling Section:**")

    # ************COLUMN PLACEHOLDERS *****************************************************
    # labelling_section_clusters_button_col,_,start_labelling_button_col=st.columns([1,3,1])
    all_task_button_col, _, labelled_task_button_col, _, queue_button_col, _, start_labelling_button_col = st.columns([
        2, 0.5, 3, 0.5, 2, 5, 3])
    # ************COLUMN PLACEHOLDERS *****************************************************

    # TODO #90 Add Pagination for Labelling Section
    labelling_page = {
        LabellingPagination.AllTask: all_task_table,
        LabellingPagination.Labelled: labelled_table,
        LabellingPagination.Queue: no_labelled,
        LabellingPagination.Editor: None,
        LabellingPagination.Performance: None
    }

    # >>>> INSTANTIATE LABELLING PAGINATION
    if 'labelling_pagination' not in session_state:
        session_state.labelling_pagination = LabellingPagination.AllTask

    # >>>> PAGINATION BUTTONS  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def to_labelling_section(section_name: IntEnum):
        session_state.labelling_pagination = section_name

    with all_task_button_col:
        st.button("All Task", key='all_task_button', on_click=to_labelling_section, args=(
            LabellingPagination.AllTask,))

    with labelled_task_button_col:
        st.button("Labelled Task", key='labelled_task_button',
                  on_click=to_labelling_section, args=(LabellingPagination.Labelled,))

    with queue_button_col:
        st.button("Queue", key='queue_button', on_click=to_labelling_section, args=(
            LabellingPagination.Queue,))

    with start_labelling_button_col:
        st.button("Start Labelling", key='start_labelling_button',
                  on_click=to_labelling_section, args=(LabellingPagination.Editor,))
    # >>>> PAGINATION BUTTONS  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    project_id = session_state.project.id
    all_task, all_task_column_names = Project.query_all_task(project_id,
                                                             return_dict=True, for_data_table=True)

    # >>>> MAIN FUNCTION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    log_info(f"Navigator: {session_state.labelling_pagination}")
    labelling_page[session_state.labelling_pagination](all_task)


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        index()

    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
