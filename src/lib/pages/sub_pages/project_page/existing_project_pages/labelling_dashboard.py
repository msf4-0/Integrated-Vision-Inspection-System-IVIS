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
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

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


def _all_task_table(all_task: List):
    # **************** DATA TABLE COLUMN CONFIG *********************************************************
    all_task_columns = [
        {
            'field': "id",
            'headerName': "ID",
            'headerAlign': "center",
            'align': "center",
            'flex': 50,
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
            'flex': 150,
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

    # >>>> ALL TASK TABLE >>>>>>>>>>>>>>>>>>>>>>

    data_table(all_task, all_task_columns,
               checkbox=False, key='all_task_table_key')


def dashboard():
    # TODO 80 Add Labelling section
    # 1. All Data Table
    # 2. Queue
    # 3. Labelled Table** Is it redundant to All Data Table????

    # Query all task +user full name(concat)+ dataset name + annotation status
    project_id = session_state.project.id
    all_task, all_task_column_names = Project.query_all_task(project_id,
                                                             return_dict=True, for_data_table=True)

    # >>>> ALL TASK TABLE >>>>>>>>>>>>>>>>>>>>>>
    _all_task_table(all_task)


def index():
    RELEASE = False

    # ****************** TEST ******************************
    if not RELEASE:
        log_info("At Exisiting Project Dashboard INDEX")

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

        

    # TODO #90 Add Pagination for Labelling Section
    labelling_page = {
        LabellingPagination.Dashboard: dashboard,
        LabellingPagination.Labelled: None,
        LabellingPagination.Queue: None,
        LabellingPagination.Editor: None,
        LabellingPagination.Performance:None
    } 

    if 'labelling_pagination' not in session_state:
        session_state.labelling_pagination = LabellingPagination.Dashboard


    log_info(f"Navigator: {session_state.labelling_pagination}")
    labelling_page[session_state.labelling_pagination]()

if __name__ == "__main__":
    if st._is_running_with_streamlit:
        index()

    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
