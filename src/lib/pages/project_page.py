"""
Title: Project Page
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
layout = 'wide'
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)


# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>
SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
# TEST_MODULE_PATH = SRC / "test" / "test_page" / "module"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
from data_manager.database_manager import init_connection
from project.project_management import NewProject, ProjectPagination
from pages.sub_pages.dataset_page.new_dataset import new_dataset
from pages.sub_pages.project_page import new_project
# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])
PAGE_OPTIONS = {"Dataset", "Project", "Deployment"}

# <<<< Variable Declaration <<<<
chdir_root()  # change to root directory

with st.sidebar.container():
    st.image("resources/MSF-logo.gif", use_column_width=True)

    st.title("Integrated Vision Inspection System", anchor='title')
    st.header(
        "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
    st.markdown("""___""")
    st.radio("", options=PAGE_OPTIONS, key="all_pages")

navigator = st.sidebar.empty()


def dashboard():
    st.write(f"# Nothing")


def main():

    # dataset_page = {
    #     DatasetPagination.Dashboard: dashboard,
    #     DatasetPagination.New: new_dataset.show
    # }
    # if 'dataset_pagination' not in session_state:
    #     session_state.dataset_pagination = DatasetPagination.Dashboard

    # project_page_options = ("Dashboard", "Create New Dataset")

    # def dataset_page_navigator():
    #     session_state.dataset_pagination = project_page_options.index(
    #         session_state.dataset_page_navigator_radio)

    # if "dataset_page_navigator_radio" in session_state:
    #     del session_state.dataset_page_navigator_radio

    # with st.sidebar.expander("Dataset", expanded=True):
    #     st.radio("", options=project_page_options,
    #              index=session_state.dataset_pagination, on_change=dataset_page_navigator, key="dataset_page_navigator_radio")

    # dataset_page[session_state.dataset_pagination]()

    project_page = {
        ProjectPagination.Dashboard: dashboard,
        ProjectPagination.New: new_project.index,
        ProjectPagination.Existing: None,
        ProjectPagination.NewDataset: new_dataset
    }

    if 'project_pagination' not in session_state:
        session_state.project_pagination = ProjectPagination.Dashboard

    if 'project_status' not in session_state:
        session_state.project_status = None

    project_page_options = ("Dashboard", "Create New Project")

    def project_page_navigator():

        NewProject.reset_new_project_page()

        session_state.project_pagination = project_page_options.index(
            session_state.project_page_navigator_radio)

        if "project_page_navigator_radio" in session_state:
            del session_state.project_page_navigator_radio

    with navigator.expander("Project", expanded=True):
        st.radio("", options=project_page_options,
                 index=session_state.project_pagination, on_change=project_page_navigator, key="project_page_navigator_radio")

    project_page[session_state.project_pagination]()


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
