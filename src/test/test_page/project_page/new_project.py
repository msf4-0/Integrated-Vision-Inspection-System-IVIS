"""
Title: New Project Page
Date: 5/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path

from streamlit.state.session_state import SessionState
# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[3]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
TEST_MODULE_PATH = SRC / "test" / "test_page" / "module"

for path in sys.path:
    if str(LIB_PATH) not in sys.path:
        sys.path.insert(0, str(LIB_PATH))  # ./lib
    else:
        pass

    if str(TEST_MODULE_PATH) not in sys.path:
        sys.path.insert(0, str(TEST_MODULE_PATH))
    else:
        pass

from path_desc import chdir_root
from code_generator import get_random_string
from core.utils.log import std_log  # logger

# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
from time import sleep
import logging
import psycopg2

import streamlit as st
# DEFINE Web APP page configuration
layout = 'wide'
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)


@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
def init_connection():
    return psycopg2.connect(**st.secrets["postgres"])
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<


# >>>> Variable Declaration
new_project = {}  # store
place = {}
DEPLOYMENT_TYPE = ("", "Image Classification", "Object Detection with Bounding Boxes",
                   "Semantic Segmentation with Polygons", "Semantic Segmentation with Masks")

# >>>> TODO: query from Database
DATASET_LIST = ["Hello"]


def show():

    chdir_root()  # change to root directory

    # >>>> START >>>>
    with st.sidebar.beta_container():

        st.image("resources/MSF-logo.gif", use_column_width=True)
    # with st.beta_container():
        st.title("Integrated Vision Inspection System", anchor='title')

        st.header(
            "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
        st.markdown("""___""")

    # <<<< START <<<<

    conn = init_connection()  # initialise connection to Database

    if "current_page" not in st.session_state:  # KIV
        st.session_state.current_page = "All Projects"
        st.session_state.previous_page = "All Projects"

    # >>>> Project Sidebar >>>>
    project_page_options = ("All Projects", "New Project")
    with st.sidebar.beta_expander("Project Page", expanded=True):
        st.session_state.current_page = st.radio("project_page_select", options=project_page_options,
                                                 index=0)
    # <<<< Project Sidebar <<<<

    # >>>> New Project MAIN >>>>
    # Page title
    st.write("# __Add New Project__")
    st.markdown("___")

    # Session State store new project ID
    if 'project_id' not in st.session_state:
        # set random project ID before getting actual from Database
        st.session_state.project_id = get_random_string(length=8)
    # reference to project ID session state
    new_project["id"] = st.session_state.project_id

    # right-align the project ID relative to the page
    id_blank, id_right = st.beta_columns([3, 1])
    id_right.write(f"### __Project ID:__ {new_project['id']}")

    create_project_place = st.empty()
    # if layout == 'wide':
    #     col1, col2, col3 = create_project_place.beta_columns([1, 3, 1])
    # else:
    #     col2 = create_project_place
    with create_project_place.beta_container():
        st.write("## __Project Information :__")

        new_project["title"] = st.text_input(
            "Project Title", key="title", help="Enter the name of the project")
        place["title"] = st.empty()

        # **** Optional ****
        new_project["desc"] = st.text_area(
            "Description (Optional)", key="desc", help="Enter the description of the project")
        place["title"] = st.empty()

        new_project["deployment_type"] = st.selectbox(
            "Deployment Type", key="deployment_type", options=DEPLOYMENT_TYPE, format_func=lambda x: 'Select an option' if x == '' else x, help="Select the type of deployment of the project")
        place["deployment_type"] = st.empty()

        DATASET_LIST.insert(0, "")

        # **** Optional ****
        st.write("## __Dataset :__")
        new_project["dataset"] = st.multiselect(
            "Dataset", key="dataset", options=DATASET_LIST, format_func=lambda x: 'Select an option' if x == '' else x, help="Select the type of deployment of the project")
        place["dataset"] = st.empty()

        # **** Optional ****
        st.write("## __Image Augmentation :__")
        new_project["augmentation"] = st.multiselect(
            "Augmentation", key="augmentation", options=DATASET_LIST, format_func=lambda x: 'Select an option' if x == '' else x, help="Select the type of deployment of the project")
        place["augmentation"] = st.empty()

        st.write(new_project)


if __name__ == "__main__":
    show()
