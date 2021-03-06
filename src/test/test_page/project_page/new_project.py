"""
Title: New Project Page
Date: 5/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
import pandas as pd


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
import numpy as np  # TEMP for table viz

# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
from time import sleep
import logging
import psycopg2

import streamlit as st
from streamlit import session_state as SessionState
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
                   "Semantic Segmentation with Polygons")

# >>>> TODO: query from Database
DATASET_LIST = list('abcdefghijabcdefghij')


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

    if "current_page" not in SessionState:  # KIV
        SessionState.current_page = "All Projects"
        SessionState.previous_page = "All Projects"

    if "new_project" not in SessionState:
        SessionState.new_project = {}
        SessionState.new_project["id"] = get_random_string(length=8)
        # set random project ID before getting actual from Database

    # >>>> Project Sidebar >>>>
    project_page_options = ("All Projects", "New Project")
    with st.sidebar.beta_expander("Project Page", expanded=True):
        SessionState.current_page = st.radio("project_page_select", options=project_page_options,
                                             index=0)
    # <<<< Project Sidebar <<<<

    # >>>> New Project MAIN >>>>
    # Page title
    st.write("# __Add New Project__")
    st.markdown("___")

    # right-align the project ID relative to the page
    id_blank, id_right = st.beta_columns([3, 1])
    id_right.write(f"### __Project ID:__ {SessionState.new_project['id']}")

    create_project_place = st.empty()
    # if layout == 'wide':
    #     col1, col2, col3 = create_project_place.beta_columns([1, 3, 1])
    # else:
    #     col2 = create_project_place
    with create_project_place.beta_container():
        st.write("## __Project Information :__")

        SessionState.new_project["title"] = st.text_input(
            "Project Title", key="title", help="Enter the name of the project")
        place["title"] = st.empty()

        # **** Project Description (Optional) ****
        SessionState.new_project["desc"] = st.text_area(
            "Description (Optional)", key="desc", help="Enter the description of the project")
        place["title"] = st.empty()

        SessionState.new_project["deployment_type"] = st.selectbox(
            "Deployment Type", key="deployment_type", options=DEPLOYMENT_TYPE, format_func=lambda x: 'Select an option' if x == '' else x, help="Select the type of deployment of the project")
        place["deployment_type"] = st.empty()

        # **** Dataset (Optional) ****

        # include options to create new dataset on this page
        # create 2 columns for "New Data Button"
        st.write("## __Dataset :__")

        data_left, data_right = st.beta_columns(2)
        # >>>> Right Column to select dataset >>>>
        with data_right:
            SessionState.new_project["dataset"] = st.multiselect(
                "Dataset List", key="dataset", options=DATASET_LIST, format_func=lambda x: 'Select an option' if x == '' else x, help="Select the type of deployment of the project")

            # Button to create new dataset
            new_data_button = st.button("Create New Dataset")

            # print choosen dataset
            st.write("### Dataset choosen:")
            if len(SessionState.new_project["dataset"]) > 0:
                for idx, data in enumerate(SessionState.new_project["dataset"]):
                    st.write(f"{idx+1}. {data}")
            elif len(SessionState.new_project["dataset"]) == 0:
                st.info("No dataset selected")
        # <<<< Right Column to select dataset <<<<

        # >>>> Left Column to show full list of dataset and selection >>>>
        if "dataset_page" not in SessionState:
            SessionState.dataset_page = 0

        def next_page():
            SessionState.dataset_page += 1

        def prev_page():
            SessionState.dataset_page -= 1

        with data_left:
            start = 10 * SessionState.dataset_page
            end = start + 10

            df = pd.DataFrame(np.random.rand(20, 4), columns=(
                'col{}'.format(i) for i in range(4)), index=DATASET_LIST)

            def highlight_row(x, selections):

                if x.name in selections:

                    return ['background-color: #90a4ae'] * len(x)
                else:
                    return ['background-color: '] * len(x)
            df_slice = df.iloc[start:end]

            # >>>>DATAFRAME
            st.dataframe(df_slice.style.apply(
                highlight_row, selections=SessionState.new_project["dataset"], axis=1))
        # <<<< Left Column to show full list of dataset and selection <<<<

        # >>>> Dataset Pagination >>>>
        col1, col2, col3, _ = st.beta_columns([0.15, 0.2, 0.15, 0.5])
        num_dataset_per_page = 10
        num_dataset_page = len(DATASET_LIST) // num_dataset_per_page
        # st.write(num_dataset_page)
        if num_dataset_page > 1:
            if SessionState.dataset_page < num_dataset_page:
                col3.button(">", on_click=next_page)
            else:
                col3.write("")  # this makes the empty column show up on mobile

            if SessionState.dataset_page > 0:
                col1.button("<", on_click=prev_page)
            else:
                col1.write("")  # this makes the empty column show up on mobile

        col2.write(
            f"Page {1+SessionState.dataset_page} of {num_dataset_page}")
        # <<<< Dataset Pagination <<<<
        place["dataset"] = st.empty()  # TODO :KIV

        # **** Image Augmentation (Optional) ****
        st.write("## __Image Augmentation :__")
        SessionState.new_project["augmentation"] = st.multiselect(
            "Augmentation List", key="augmentation", options=DATASET_LIST, format_func=lambda x: 'Select an option' if x == '' else x, help="Select the type of deployment of the project")
        place["augmentation"] = st.empty()

        # **** Training Parameters (Optional) ****
        st.write("## __Training Parameters :__")
        SessionState.new_project["training_param"] = st.multiselect(
            "Training Parameters", key="training_param", options=DATASET_LIST, format_func=lambda x: 'Select an option' if x == '' else x, help="Select the type of deployment of the project")
        place["augmentation"] = st.empty()

        # **** Submit Button ****
        col1, col2 = st.beta_columns([3, 0.5])
        submit_button = col2.button("Submit", key="submit")

        st.write(SessionState.new_project)


if __name__ == "__main__":
    show()
