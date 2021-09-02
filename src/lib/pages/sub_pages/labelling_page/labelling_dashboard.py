"""
Title: Labelling Dashboard
Date: 19/8/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from copy import deepcopy
from enum import IntEnum
from pathlib import Path
from time import sleep
from typing import List

import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state as session_state

# DEFINE Web APP page configuration
layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

from data_table import data_table

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

from annotation.annotation_management import (LabellingPagination,
                                              reset_editor_page)
from core.utils.log import log_error, log_info  # logger
from data_editor.data_labelling import editor
from data_editor.editor_config import editor_config
from data_manager.database_manager import init_connection
from path_desc import chdir_root
from project.project_management import (ExistingProjectPagination, Project,
                                        ProjectPermission)
from annotation.annotation_management import Task

# NOTE Temp
from user.user_management import User
# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


# >>>> Variable Declaration >>>>
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

# ************************* TO LABELLING CALLBACK ***********************************
def to_labelling_editor_page(section_enum: IntEnum):
    labelling_section_data_table_dict = {
        LabellingPagination.AllTask: 'all_task_table_key',
        LabellingPagination.Labelled: 'labelled_task_table_key',
        LabellingPagination.Queue: 'task_queue_table_key'
    }

    labelling_section_data_table = labelling_section_data_table_dict[section_enum]

    # Get state value from the respective Data Table
    session_state.data_selection = deepcopy(
        session_state[labelling_section_data_table])
    # Change pagination to Editor
    session_state.labelling_pagination = LabellingPagination.Editor
    # Reset selection at Data Table
    if labelling_section_data_table in session_state:
        del session_state[labelling_section_data_table]
    if "new_annotation_flag" not in session_state:
        session_state.new_annotation_flag = 0
    else:
        session_state.new_annotation_flag = 0
    log_info(
        f"Data selected, entering editor for data {session_state.data_selection} for {labelling_section_data_table}")
    st.experimental_rerun()

# ************************* TO LABELLING CALLBACK ***********************************


def no_labelled(all_task, labelled_task_dict, task_queue_dict):

    # >>>> TASK QUEUE TABLE >>>>>>>>>>>>>>>>>>>>>>
    # >>>> All task table placeholders

    task_queue_table_col1, task_queue_table_col2 = st.columns([3, 0.75])
    length_of_queue = len(task_queue_dict)
    total_task_length = len(all_task)

    task_queue_table_col1.write(f"### Task Queue")
    task_queue_table_col2.write(
        f"### Total remaining data: {length_of_queue}/{total_task_length}")
    data_table(task_queue_dict, all_task_columns,
               checkbox=False, key='task_queue_table_key', on_change=to_labelling_editor_page, args=(LabellingPagination.Queue,))


def labelled_table(all_task, labelled_task_dict, task_queue_dict):

    # >>>> LABELLED TASK TABLE >>>>>>>>>>>>>>>>>>>>>>

    # >>>> All task table placeholders
    labelled_task_table_col1, labelled_task_table_col2 = st.columns([3, 0.75])

    length_of_labelled = len(labelled_task_dict)
    length_of_remaining = len(all_task) - len(labelled_task_dict)

    labelled_task_table_col1.write(f"### Labelled Task")
    labelled_task_table_col2.write(
        f"### Total labelled data: {length_of_labelled}/{len(all_task)}")
    data_table(labelled_task_dict, all_task_columns,
               checkbox=False, key='labelled_task_table_key', on_change=to_labelling_editor_page, args=(LabellingPagination.Labelled,))


def all_task_table(all_task, labelled_task_dict, task_queue_dict):

    # >>>> ALL TASK TABLE >>>>>>>>>>>>>>>>>>>>>>

    # >>>> All task table placeholders
    all_task_table_col1, all_task_table_col2, all_task_table_col3, all_task_table_col4 = st.columns([
                                                                                                    2, 0.3, 0.5, 0.5])
    _, all_task_table_bottom_col1, all_task_table_bottom_col2 = st.columns([
                                                                           3, 0.75, 0.75])

    all_task_table_col1.write(f"### All Task")
    all_task_table_col2.write(f"### Total data: {len(all_task)}")
    all_task_table_col3.write(
        f"### Total labelled data: {len(labelled_task_dict)}")
    all_task_table_col4.write(
        f"### Total remaining data: {len(task_queue_dict)}")

    data_table(all_task, all_task_columns,
               checkbox=False, key='all_task_table_key', on_change=to_labelling_editor_page, args=(LabellingPagination.AllTask,))


def index():

    RELEASE = True

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
        if 'user' not in session_state:
            session_state.user = User(1)
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

    labelling_page = {
        LabellingPagination.AllTask: all_task_table,
        LabellingPagination.Labelled: labelled_table,
        LabellingPagination.Queue: no_labelled,
        LabellingPagination.Editor: editor,
        LabellingPagination.Performance: None
    }

    # >>>> INSTANTIATE LABELLING PAGINATION
    if 'labelling_pagination' not in session_state:
        session_state.labelling_pagination = LabellingPagination.AllTask
    if 'data_selection' not in session_state:
        # state store for Data Table
        session_state.data_selection = []
    # >>>> PAGINATION BUTTONS  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def to_labelling_section(section_name: IntEnum):
        session_state.labelling_pagination = section_name
        reset_editor_page()

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
    all_task, all_task_column_names = Task.query_all_task(project_id,
                                                          return_dict=True, for_data_table=True)
    labelled_task_dict = Task.get_labelled_task(all_task, True)
    task_queue_dict = Task.get_labelled_task(all_task, False)
    # >>>> MAIN FUNCTION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    log_info(f"Navigator: {session_state.labelling_pagination}")

    if session_state.labelling_pagination == LabellingPagination.Editor:
        # if "new_annotation_flag" not in session_state:
        #     session_state.new_annotation_flag = 0
        # else:
        #     session_state.new_annotation_flag = 0
        labelling_page[session_state.labelling_pagination](
            session_state.data_selection)

    else:
        labelling_page[session_state.labelling_pagination](
            all_task, labelled_task_dict, task_queue_dict)


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        index()

    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
