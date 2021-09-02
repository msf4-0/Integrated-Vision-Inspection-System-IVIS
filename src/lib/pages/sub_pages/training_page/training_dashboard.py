"""
Title: Training Dashboard
Date: 30/8/2021
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
layout = 'wide'
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
from data_manager.database_manager import init_connection
from project.project_management import ExistingProjectPagination, ProjectPermission, Project
from training.training_management import TrainingPagination, Training
from pages.sub_pages.training_page import new_training
# >>>> TEMP
from user.user_management import User
# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


# >>>> Variable Declaration >>>>

place = {}
DEPLOYMENT_TYPE = ("", "Image Classification", "Object Detection with Bounding Boxes",
                   "Semantic Segmentation with Polygons", "Semantic Segmentation with Masks")


chdir_root()  # change to root directory


def dashboard():
    log_info(f"Top of Training Dashboard")
    st.write(f"### Dashboard")

    # ******** SESSION STATE *********************************************************
    if "project_training_table" not in session_state:
        session_state.project_training_table = None
    # ******** SESSION STATE *********************************************************

    # ************COLUMN PLACEHOLDERS *****************************************************
    create_new_training_button_col1 = st.empty()

    # ************COLUMN PLACEHOLDERS *****************************************************

    # ***************** CREATE NEW PROJECT BUTTON *********************************************************
    def to_new_training_page():

        session_state.training_pagination = TrainingPagination.New

        if "project_training_table" in session_state:
            del session_state.project_training_table

    create_new_training_button_col1.button(
        "Create New Training Session", key='create_new_training_from_training_dashboard',
        on_click=to_new_training_page, help="Create a new training session")
    # ***************** CREATE NEW PROJECT BUTTON *********************************************************


def index():
    RELEASE = False
    log_info("At Training Dashboard INDEX")
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
        project_id_tmp = 43
        log_info(f"Entering Project {project_id_tmp}")

        session_state.append_project_flag = ProjectPermission.ViewOnly

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
    st.write(f"## **Training Section:**")
    # ************************ EXISTING PROJECT PAGINATION *************************
    training_page = {
        TrainingPagination.Dashboard: dashboard,
        TrainingPagination.New: new_training.new_training_page,
        TrainingPagination.Existing: None,
        TrainingPagination.NewModel: None
    }

    if 'training_pagination' not in session_state:
        session_state.training_pagination = TrainingPagination.Dashboard

    # >>>> RETURN TO ENTRY PAGE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    training_dashboard_back_button_place = st.empty()

    if session_state.training_pagination != TrainingPagination.Dashboard:

        def to_training_dashboard_page():
            # TODO #133 Add New Training Reset
            session_state.training_pagination = TrainingPagination.Dashboard

        training_dashboard_back_button_place.button("Back to Training Dashboard", key="back_to_training_dashboard_page",
                                                    on_click=to_training_dashboard_page)

    else:
        training_dashboard_back_button_place.empty()

    log_info(
        f"Entering Training Page:{session_state.training_pagination}")

    # TODO #132 Add reset to training session state
    # >>>> MAIN FUNCTION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    training_page[session_state.training_pagination]()


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        index()

    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
