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
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


# >>>> Variable Declaration >>>>
new_project = {}  # store
place = {}
DEPLOYMENT_TYPE = ("", "Image Classification", "Object Detection with Bounding Boxes",
                   "Semantic Segmentation with Polygons", "Semantic Segmentation with Masks")


chdir_root()  # change to root directory


def existing_project():
    st.write("# Existing")
    # TODO #79 Add dashboard to show types of labels and number of datasets
    # TODO #80 Add Labelling interface
    st.write(vars(session_state.project))


def index():

    existing_project()


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        index()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
