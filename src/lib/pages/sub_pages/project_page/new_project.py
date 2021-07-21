"""
Title: New Project Page
Date: 5/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
import psycopg2
import pandas as pd
import numpy as np  # TEMP for table viz
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
# TEST_MODULE_PATH = SRC / "test" / "test_page" / "module"

for path in sys.path:
    if str(LIB_PATH) not in sys.path:
        sys.path.insert(0, str(LIB_PATH))  # ./lib
    else:
        pass

from path_desc import chdir_root
from core.utils.code_generator import get_random_string
from core.utils.log import log_info, log_error  # logger
import numpy as np  # TEMP for table viz
from project.project_management import NewProject
from data_manager.database_manager import init_connection, db_fetchone
# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


# >>>> Variable Declaration >>>>
new_project = {}  # store
place = {}
DEPLOYMENT_TYPE = ("", "Image Classification", "Object Detection with Bounding Boxes",
                   "Semantic Segmentation with Polygons", "Semantic Segmentation with Masks")


class AnnotationType(IntEnum):
    Image_Classification = 1
    BBox = 2
    Polygons = 3
    Masks = 4


# >>>> TODO: query from Database
DATASET_LIST = list('abcdefghijabcdefghij')


def show():

    chdir_root()  # change to root directory

    with st.sidebar.beta_container():

        st.image("resources/MSF-logo.gif", use_column_width=True)
    # with st.beta_container():
        st.title("Integrated Vision Inspection System", anchor='title')

        st.header(
            "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
        st.markdown("""___""")

    # ******** SESSION STATE ********

    if "current_page" not in session_state:  # KIV
        session_state.current_page = "All Projects"
        session_state.previous_page = "All Projects"

    if "new_project" not in session_state:
        session_state.new_project = NewProject(get_random_string(length=8))
        # set random project ID before getting actual from Database
        session_state.dataset_page = 0
    # ******** SESSION STATE ********

    # >>>> PROJECT SIDEBAR >>>>
    project_page_options = ("All Projects", "New Project")
    with st.sidebar.beta_expander("Project Page", expanded=True):
        session_state.current_page = st.radio("project_page_select", options=project_page_options,
                                              index=0)
    # <<<< PROJECT SIDEBAR <<<<

# >>>> New Project INFO >>>>
    # Page title
    st.write("# __Add New Project__")
    st.markdown("___")

    # right-align the project ID relative to the page
    id_blank, id_right = st.beta_columns([3, 1])
    id_right.write(
        f"### __Project ID:__ {session_state.new_project.id}")

    create_project_place = st.empty()
    # if layout == 'wide':
    outercol1, outercol2, outercol3 = st.beta_columns([1.5, 3.5, 0.5])
    # else:
    #     col2 = create_project_place
    # with create_project_place.beta_container():
    outercol1.write("## __Project Information :__")

    # >>>> CHECK IF NAME EXISTS CALLBACK >>>>
    def check_if_name_exist(field_placeholder, conn):
        context = ['name', session_state.name]
        if session_state.name:
            if session_state.new_project.check_if_exist(context, conn):
                field_placeholder['name'].error(
                    f"Dataset name used. Please enter a new name")
                sleep(1)
                log_error(f"Dataset name used. Please enter a new name")
            else:
                session_state.new_project.name = session_state.name
                log_info(f"Dataset name fresh and ready to rumble")

    outercol2.text_input(
        "Project Title", key="name", help="Enter the name of the project", on_change=check_if_name_exist, args=(place, conn,))
    place["name"] = outercol2.empty()

    # **** Project Description (Optional) ****
    description = outercol2.text_area(
        "Description (Optional)", key="desc", help="Enter the description of the project")
    if description:
        session_state.new_project.desc = description
    else:
        pass

    deployment_type = outercol2.selectbox(
        "Deployment Type", key="deployment_type", options=DEPLOYMENT_TYPE, format_func=lambda x: 'Select an option' if x == '' else x, help="Select the type of deployment of the project")

    if deployment_type is not None:
        session_state.new_project.deployment_type = deployment_type
        session_state.new_project.query_deployment_id()
        st.write(session_state.new_project.deployment_id)

    else:
        pass

    place["deployment_type"] = outercol2.empty()

# <<<<<<<< New Project INFO <<<<<<<<

# >>>>>>>> Choose Dataset >>>>>>>>

    # include options to create new dataset on this page
    # create 2 columns for "New Data Button"
    outercol1, _, _ = st.beta_columns([1.5, 3.5, 0.5])

    outercol1.write("## __Dataset :__")
    outercol1, outercol2, outercol3 = st.beta_columns([1.5, 2, 2])

    data_left, data_right = st.beta_columns(2)
    # >>>> Right Column to select dataset >>>>
    with outercol3:
        session_state.new_project.dataset = st.multiselect(
            "Dataset List", key="dataset", options=DATASET_LIST, format_func=lambda x: 'Select an option' if x == '' else x, help="Select the type of deployment of the project")

        # Button to create new dataset
        new_data_button = st.button("Create New Dataset")

        # print choosen dataset
        st.write("### Dataset choosen:")
        if len(session_state.new_project.dataset) > 0:
            for idx, data in enumerate(session_state.new_project.dataset):
                st.write(f"{idx+1}. {data}")
        elif len(session_state.new_project.dataset) == 0:
            st.info("No dataset selected")
    # <<<< Right Column to select dataset <<<<

    # >>>> Left Column to show full list of dataset and selection >>>>
    if "dataset_page" not in session_state:
        session_state.dataset_page = 0

    def next_page():
        session_state.dataset_page += 1

    def prev_page():
        session_state.dataset_page -= 1

    with outercol2:
        start = 10 * session_state.dataset_page
        end = start + 10

        df = session_state.new_project.create_dataset_dataframe()
        

        def highlight_row(x, selections):

            if x.name in selections:

                return ['background-color: #90a4ae'] * len(x)
            else:
                return ['background-color: '] * len(x)
        df_slice = df.iloc[start:end]
        styler = df_slice.style.format(
                {
                    "Date/Time": lambda t: t.strftime('%Y-%m-%d %H:%M:%S')

                }
            )

        # >>>>DATAFRAME
        st.table(styler.apply(
            highlight_row, selections=session_state.new_project.dataset, axis=1))
    # <<<< Left Column to show full list of dataset and selection <<<<

    # >>>> Dataset Pagination >>>>
    col1, col2, col3, _ = st.beta_columns([0.15, 0.2, 0.15, 0.5])
    num_dataset_per_page = 10
    num_dataset_page = len(DATASET_LIST) // num_dataset_per_page
    # st.write(num_dataset_page)
    if num_dataset_page > 1:
        if session_state.dataset_page < num_dataset_page:
            col3.button(">", on_click=next_page)
        else:
            col3.write("")  # this makes the empty column show up on mobile

        if session_state.dataset_page > 0:
            col1.button("<", on_click=prev_page)
        else:
            col1.write("")  # this makes the empty column show up on mobile

    col2.write(
        f"Page {1+session_state.dataset_page} of {num_dataset_page}")
    # <<<< Dataset Pagination <<<<
    place["dataset"] = st.empty()  # TODO :KIV

    # # **** Image Augmentation (Optional) ****
    # st.write("## __Image Augmentation :__")
    # session_state.new_project["augmentation"] = st.multiselect(
    #     "Augmentation List", key="augmentation", options=DATASET_LIST, format_func=lambda x: 'Select an option' if x == '' else x, help="Select the type of deployment of the project")
    # place["augmentation"] = st.empty()

    # # **** Training Parameters (Optional) ****
    # st.write("## __Training Parameters :__")
    # session_state.new_project["training_param"] = st.multiselect(
    #     "Training Parameters", key="training_param", options=DATASET_LIST, format_func=lambda x: 'Select an option' if x == '' else x, help="Select the type of deployment of the project")
    # place["augmentation"] = st.empty()

    # **** Submit Button ****
    col1, col2 = st.beta_columns([3, 0.5])
    submit_button = col2.button("Submit", key="submit")

    st.write(vars(session_state.new_project))


def main():
    show()


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
