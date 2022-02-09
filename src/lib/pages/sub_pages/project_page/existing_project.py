"""
Title: New Project Page
Date: 5/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state
from deployment.deployment_management import Deployment
from pages.sub_pages.training_page import training_dashboard

# DEFINE Web APP page configuration
# layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
# LIB_PATH = SRC / "lib"

# if str(LIB_PATH) not in sys.path:
#     sys.path.insert(0, str(LIB_PATH))  # ./lib
# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

from path_desc import chdir_root
from core.utils.log import logger  # logger
from data_manager.database_manager import init_connection
from project.project_management import ExistingProjectPagination, ProjectPermission, Project

from pages.sub_pages.dataset_page.new_dataset import new_dataset
from pages.sub_pages.project_page.existing_project_pages import existing_project_dashboard, settings
from pages.sub_pages.labelling_page import labelling_dashboard

from annotation.annotation_management import reset_editor_page
from training.training_management import NewTraining, Training
from pages import deployment_navigation
# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


# >>>> Variable Declaration >>>>
new_project = {}  # store
place = {}
DEPLOYMENT_TYPE = ("", "Image Classification", "Object Detection with Bounding Boxes",
                   "Semantic Segmentation with Polygons")


def index():
    chdir_root()  # change to root directory

    RELEASE = True
    logger.debug("Navigator: At Existing Project Dashboard INDEX")
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
        project_id_tmp = 7
        logger.info(f"Entering Project {project_id_tmp}")

        session_state.append_project_flag = ProjectPermission.ViewOnly

        if "project" not in session_state:
            session_state.project = Project(project_id_tmp)
            logger.info("Inside")

        # else:
        #     session_state.project = Project(project_id_tmp)

    # ************************ EXISTING PROJECT PAGINATION *************************
    existing_project_page = {
        ExistingProjectPagination.Dashboard: existing_project_dashboard.index,
        ExistingProjectPagination.Labelling: labelling_dashboard.index,
        ExistingProjectPagination.Training: training_dashboard.index,
        # ExistingProjectPagination.Models: None,
        ExistingProjectPagination.Deployment: deployment_navigation.index,
        ExistingProjectPagination.Settings: settings.index
    }

    # ****************************** HEADER **********************************************
    st.write(f"# Project: {session_state.project.name}")

    project_description = session_state.project.desc if session_state.project.desc is not None else " "
    st.write(f"{project_description}")

    st.markdown("""___""")
    # ****************************** HEADER **********************************************

    if 'existing_project_pagination' not in session_state:
        session_state.existing_project_pagination = ExistingProjectPagination.Dashboard

    logger.debug(f"Entering Project {session_state.project.id}: "
                 f"{session_state.existing_project_pagination = }")

    session_state.append_project_flag = ProjectPermission.ViewOnly
    # >>>> Pagination RADIO >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # note that this must follow the exact order of ExistingProjectPagination,
    # check the callback function to understand why
    existing_project_page_options = (
        "Overview", "Labelling", "Training", "Deployment", "Settings")

    # >>>> CALLBACK for RADIO >>>>
    def existing_project_page_navigator():

        navigation_selected = session_state.existing_project_page_navigator_radio
        session_state.existing_project_pagination = ExistingProjectPagination.from_string(
            navigation_selected)
        # navigation_selected_idx = existing_project_page_options.index(
        #     navigation_selected)
        # session_state.existing_project_pagination = navigation_selected_idx

        # NOTE: TO RESET SUB-PAGES AFTER EXIT
        # reset all pages except for the currently selected one
        if navigation_selected == "Overview":
            reset_editor_page()
            NewTraining.reset_new_training_page()
            Training.reset_training_page()
            Project.reset_settings_page()
            Deployment.reset_deployment_page()
        elif navigation_selected == "Labelling":
            Project.reset_dashboard_page()
            NewTraining.reset_new_training_page()
            Training.reset_training_page()
            Project.reset_settings_page()
            Deployment.reset_deployment_page()
        elif navigation_selected == "Training":
            Project.reset_dashboard_page()
            reset_editor_page()
            # also reset training page when clicked on this navigation, to clean up
            # some things that were not included in `reset_editor_page()` for reasons
            # explained in that function
            Training.reset_training_page()
            Project.reset_settings_page()
            Deployment.reset_deployment_page()
        elif navigation_selected == "Deployment":
            # NOTE: not resetting here to ensure the deployment keeps running
            # even when in another page, especially for switching user
            # Deployment.reset_deployment_page()

            Project.reset_dashboard_page()
            reset_editor_page()
            NewTraining.reset_new_training_page()
            Training.reset_training_page()
            Project.reset_settings_page()
        elif navigation_selected == "Settings":
            Project.reset_dashboard_page()
            reset_editor_page()
            NewTraining.reset_new_training_page()
            Training.reset_training_page()
            Deployment.reset_deployment_page()

    if not session_state.project.datasets:
        # only show overview and settings navigation (to allow project deletion)
        available_options = ("Overview", "Settings")
        if session_state.existing_project_pagination == ExistingProjectPagination.Dashboard:
            idx = 0
        else:
            idx = 1
    else:
        available_options = existing_project_page_options
        idx = session_state.existing_project_pagination
    with st.sidebar.expander("Project Navigation", expanded=True):
        st.radio(
            "Sections", options=available_options,
            index=idx,
            on_change=existing_project_page_navigator,
            key="existing_project_page_navigator_radio")
    st.sidebar.markdown("___")
    # >>>> Pagination RADIO >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # >>>> MAIN FUNCTION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    existing_project_page[session_state.existing_project_pagination]()


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        index()

    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
