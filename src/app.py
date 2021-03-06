"""
Title: Integrated Vision Inspection System
AUthor: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import gc
import os
import sys
from pathlib import Path
from time import sleep
import os

import streamlit as st
from streamlit import session_state
# Add CLI so can run Python script directly
from streamlit import cli as stcli
import tensorflow as tf

# ************** Add sys path for modules **************
SRC = Path(__file__).parent.resolve()  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

# DEFINE Web APP page configuration
layout = 'wide'
st.set_page_config(page_title="AI4WD-IVIS",
                   page_icon="resources/shrdc-logo_no-bg.png", layout=layout)

# user-defined modules
from path_desc import chdir_root, SECRETS_PATH
from core.utils.log import logger
from database_setup import database_setup, database_direct_setup, test_database_connection


def setup():
    # to disable warning messages from OpenCV, must do this before import cv2
    if os.getenv('DEBUG', '1') != '1':
        os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
        os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

    # limit memory growth to avoid memory issues
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            # limit memory growth to avoid memory issues
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            logger.info(f"{len(gpus)} Physical GPUs, "
                        f"{len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            logger.warning(e)
    # set to False because there are still things to setup later
    session_state.setup = False

    # ********************** Connection to db **********************
    # for Docker Compose installation
    if os.environ.get("DOCKERCOMPOSE"):
        logger.debug(f"{os.environ.get('DOCKERCOMPOSE') = }")
        if not SECRETS_PATH.exists():
            # setup entire database for the first time and generate the secrets.toml file
            database_direct_setup()
        if SECRETS_PATH.exists() and test_database_connection(**st.secrets['postgres']):
            logger.debug(f"Connected with: {st.secrets = }")
        else:
            error_txt = ('There were some error creating the database config. Please '
                         'try to refresh the page, or fix the database information in '
                         'your ".env" file.')
            logger.error(error_txt)
            st.error(error_txt)
            st.stop()

    # setup database if Streamlit's secrets.toml file is not generated yet
    if not SECRETS_PATH.exists():
        # to set it to the middle of the page
        left, mid_col, right = st.columns([1, 2, 1])
        with left:
            st.write("")
        with right:
            st.write("")
        with mid_col:
            database_setup()
        st.stop()
    elif not os.environ.get("DOCKERCOMPOSE"):
        # only need to check here if its not running on Docker,
        # because we have already setup above for Docker
        try:
            logger.debug(f"{st.secrets = }")
            conn_success = test_database_connection(
                **st.secrets["postgres"])
        except Exception as e:
            logger.error(f"Error reading the secrets.toml file: {e}")
            # remove the secrets.toml file to create again in case the file is invalid
            logger.debug("Removing the secrets.toml file")
            os.remove(SECRETS_PATH)
            st.experimental_rerun()
        if not conn_success:
            st.error("Error connecting to the database! You will be redirected to create "
                     "the database configuration shortly.")
            logger.error("Error connecting to the database!")
            logger.debug("Removing the secrets.toml file")
            os.remove(SECRETS_PATH)
            sleep(1)
            st.experimental_rerun()


def main():
    if 'setup' not in session_state:
        session_state.setup = False

    if not session_state.setup:
        setup()

    # **********************************************************

    from data_manager.database_manager import init_connection
    from main_page_management import MainPagination, reset_user_management_page
    from user.user_management import AccountStatus, User, UserRole, query_all_admins, reset_login_page
    from project.project_management import NewProject, Project
    from annotation.annotation_management import reset_editor_page
    from training.training_management import NewTraining, Training
    from data_manager.dataset_management import NewDataset
    from deployment.deployment_management import Deployment
    from pages.sub_pages.user_page import create_new_user, user_management_page, user_info

    # run cached database connection
    conn = init_connection(**st.secrets["postgres"])

    # setup admin just in case there is no admin user at all
    if not session_state.setup:
        admins = query_all_admins()
        if not admins:
            # to tell the create_new_user.py page that there is no Admin user yet
            session_state.no_admin = True
            # straight away proceed to Create User page to allow the user
            # to create an Admin user if this is the first time launching the app
            session_state.main_pagination = MainPagination.CreateUser
        else:
            # done setting up
            session_state.setup = True

    # ******************* IMPORT for PAGES *******************
    from pages import login_page, project_page
    # ********************************************************

    chdir_root()  # change to root directory

    # PAGES Dictionary
    # Import as modules from "./lib/pages"
    PAGES = {
        MainPagination.CreateUser: create_new_user.show,
        MainPagination.Login: login_page.index,
        MainPagination.Projects: project_page.index,
        MainPagination.UserManagement: user_management_page.main,
        MainPagination.UserInfo: user_info.main,
        # NOTE: not using this dataset_page
        # "DATASET": dataset_page.index,
    }

    with st.sidebar.container():
        st.image("resources/MSF-logo.gif", use_column_width=True)

        st.title("AI4WD-IVIS - Integrated Vision Inspection System", anchor='title')
        st.header(
            "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
        st.markdown("___")

    def to_project_cb():
        reset_login_page()
        reset_user_management_page()
        session_state.main_pagination = MainPagination.Projects

    def to_user_management_cb():
        reset_user_management_page()
        session_state.main_pagination = MainPagination.UserManagement

    def to_user_info_cb():
        session_state.main_pagination = MainPagination.UserInfo

    def login_cb():
        session_state.main_pagination = MainPagination.Login

    def logout_cb():
        # session_state.main_pagination = MainPagination.Logout

        session_state.user.update_status(AccountStatus.LOGGED_OUT)
        session_state.user.update_logout_session_log()
        del session_state['user']
        st.success("You have logged out successfully!")
        logger.info("Logged out successfully")

        tf.keras.backend.clear_session()
        gc.collect()

        # only need to reset everything on logout
        reset_login_page()
        reset_user_management_page()
        NewProject.reset_new_project_page()
        reset_editor_page()
        NewDataset.reset_new_dataset_page()
        NewTraining.reset_new_training_page()
        Training.reset_training_page()
        Project.reset_project_page()
        Project.reset_dashboard_page()
        Project.reset_settings_page()
        Deployment.reset_deployment_page()

        session_state.main_pagination = MainPagination.Login

    RELEASE = True
    if not RELEASE:
        if 'user' not in session_state:
            session_state.user = User(1)

    if 'main_pagination' not in session_state:
        session_state.main_pagination = MainPagination.Login

    if 'user' in session_state:
        # already logged in, show "Projects" button for navigation
        if session_state.main_pagination != MainPagination.Projects:
            # only show this button if user is not in the 'Projects' page
            st.sidebar.button("Projects", key='btn_to_projects',
                              on_click=to_project_cb)

        if session_state.user.role == UserRole.Administrator:
            # also show 'User Management' btn for navigation
            st.sidebar.button("User Management", key='btn_to_user_manage',
                              on_click=to_user_management_cb)

        st.sidebar.button("User Info", key='btn_user_info',
                          on_click=to_user_info_cb)
        st.sidebar.button("Logout", key='btn_logout', on_click=logout_cb)
    else:
        # nobody is logged in, only show a Login button
        st.sidebar.button("Login", key='btn_login', on_click=login_cb)

    st.sidebar.markdown("___")

    logger.debug(f"Navigator: {session_state.main_pagination = }")
    PAGES[session_state.main_pagination]()


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        # Change to root Project Directory
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]

        sys.exit(stcli.main())
