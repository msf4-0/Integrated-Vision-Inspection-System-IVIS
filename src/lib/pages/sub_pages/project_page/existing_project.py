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
DEPLOYMENT_TYPE = ("", "Image Classification", "Object Detection with Bounding Boxes",
                   "Semantic Segmentation with Polygons", "Semantic Segmentation with Masks")


chdir_root()  # change to root directory


def dashboard():
    st.write("# Existing")
    # TODO #79 Add dashboard to show types of labels and number of datasets
    # TODO #80 Add Labelling interface
    st.write(f"## **Project Name:** {session_state.project.name}")


def index():
    RELEASE = False

    # ****************** TEST ******************************
    if not RELEASE:
        log_info("At Exisiting Project Dashboard INDEX")

        # ************************TO REMOVE************************

        # DEFINE Web APP page configuration
        layout = 'wide'
        st.set_page_config(page_title="Integrated Vision Inspection System",
                           page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)
        with st.sidebar.container():
            st.image("resources/MSF-logo.gif", use_column_width=True)
            st.title("Integrated Vision Inspection System", anchor='title')
            st.header(
                "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
            st.markdown("""___""")

        # ************************TO REMOVE************************
        project_id_tmp = 7
        log_info(f"Entering Project {project_id_tmp}")

        session_state.append_project_flag = ProjectPermission.ViewOnly

        if "project" not in session_state:
            session_state.project = Project(project_id_tmp)

        else:
            session_state.project = Project(project_id_tmp)

    # ************************ EXISTING PROJECT PAGINATION *************************
    existing_project_page = {
        ExistingProjectPagination.Dashboard: dashboard,
        ExistingProjectPagination.Labelling: None,
        ExistingProjectPagination.Training: None,
        ExistingProjectPagination.Models: None,
        ExistingProjectPagination.Export: None,
        ExistingProjectPagination.Settings: None
    }

    if 'existing_project_pagination' not in session_state:
        session_state.existing_project_pagination = ExistingProjectPagination.Dashboard

    # >>>> Pagination RADIO >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    existing_project_page_options = (
        "Overview", "Labelling", "Training", "Models", "Export", "Settings")

    # >>>> CALLBACK for RADIO >>>>
    def existing_project_page_navigator():

        session_state.existing_project_pagination = existing_project_page_options.index(
            session_state.existing_project_page_navigator_radio)

        if "dataset_page_navigator_radio" in session_state:
            del session_state.existing_project_page_navigator_radio

    with st.sidebar.expander(session_state.project.name, expanded=True):
        st.radio("", options=existing_project_page_options,
                 index=session_state.existing_project_pagination, on_change=existing_project_page_navigator, key="existing_project_page_navigator_radio")
    # >>>> Pagination RADIO >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # >>>> MAIN FUNCTION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    existing_project_page[session_state.existing_project_pagination]()


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        index()

    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
