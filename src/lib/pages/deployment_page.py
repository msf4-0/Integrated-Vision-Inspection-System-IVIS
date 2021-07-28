"""
Title: Deployment
Date: 28/7/2021
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

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
DATA_DIR = Path.home() / '.local/share/integrated-vision-inspection-system/app_media'

# TEST_MODULE_PATH = SRC / "test" / "test_page" / "module"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
import numpy as np  # TEMP for table viz
from data_manager.database_manager import init_connection, db_fetchone
from project.training_management import DeploymentType
from deployment.deployment_management import Deployment
from project.model_management import PreTrainedModel, Model, BaseModel
from core.utils.helper import get_directory_name
# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])

# >>>> Variable Declaration >>>>
new_project = {}  # store
place = {}
# DEPLOYMENT_TYPE = ("", "Image Classification", "Object Detection with Bounding Boxes",
#                    "Semantic Segmentation with Polygons", "Semantic Segmentation with Masks")


def show():

    chdir_root()  # change to root directory

    with st.sidebar.beta_container():

        st.image("resources/MSF-logo.gif", use_column_width=True)
    # with st.beta_container():
        st.title("Integrated Vision Inspection System", anchor='title')

        st.header(
            "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
        st.markdown("""___""")

    # ******** SESSION STATE ***********************************************************
    if 'deployment' not in session_state:
        session_state.deployment = Deployment()
        session_state.pt_model = PreTrainedModel()
        session_state.p_model = Model()
        session_state.model = BaseModel()

    # ******** SESSION STATE *********************************************************

    # >>>> PROJECT SIDEBAR >>>>
    # project_page_options = ("All Projects", "New Project")
    # with st.sidebar.beta_expander("Project Page", expanded=True):
    #     session_state.current_page = st.radio("project_page_select", options=project_page_options,
    #                                           index=0)
    # # <<<< PROJECT SIDEBAR <<<<

# >>>> New Project INFO >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Page title
    st.write("# Deployment")
    st.markdown("___")

# >>>> SELECT DEPLOYMENT TYPE AND MODEL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Deployment Type queried from Database so that the name can be easily updated
    DEPLOYMENT_TYPE = [
        ""] + [dt.name for dt in session_state.deployment.deployment_list]

    MODEL_TABLE = {"Pre-trained Models": "public.pre_trained_models",
                   "Project Models": "public.models"}

    # with st.form(key="deployment_config"):
    _, col1, _, col2, _, col3, _ = st.beta_columns(
        [0.2, 1, 0.2, 1, 0.2, 1, 0.2])

    # ****** DEPLOYMENT TYPE *****************************************
    with col1:

        deployment_type_selected = st.selectbox(
            "Deployment Type", options=DEPLOYMENT_TYPE, format_func=lambda x: 'Select a Deployment Type' if x == "" else x)
        session_state.deployment.name = deployment_type_selected
        # Deployment Type ID
        session_state.deployment.id = DEPLOYMENT_TYPE.index(
            deployment_type_selected)

    # ****** MODEL TYPE *****************************************
    # Pre-fetch PRE-TRAINED MODEL and PROJECT MODEL list from database
    pt_model_info, pt_model_column_names = session_state.deployment.query_model_table(
        MODEL_TABLE["Pre-trained Models"])

    # pt_model_info = [
    #     [pt.Name,pt.Model_Path] for pt in session_state.pt_model.pt_model_list if session_state.pt_model.pt_model_list]
    p_model_info, p_model_column_names = session_state.deployment.query_model_table(
        MODEL_TABLE["Project Models"])

    with col2:
        model_type = st.selectbox("Model Type", options=[
            "Pre-trained Models", "Project Models", "User Upload (KIV)"])

    # ****** PRE-TRAINED MODELS *****************************************
    with col3:
        if model_type == 'Pre-trained Models':
            model_info = pt_model_info
            model_list = [
                model.Name for model in model_info]  # TODO: can be clustered?
            # model_path = [model.Model_Path for model in model_info]

    # ****** PROJECT MODELS *****************************************
        elif model_type == 'Project Models':
            model_info = p_model_info
            # TODO: can be clustered?
            model_list = [model.Name for model in model_info]
            # model_path = [model.Model_Path for model in model_info]
        else:
            model_list = []

        model_list.insert(0, "")
        model_selected = st.selectbox(
            "Model List", options=model_list, format_func=lambda x: 'Select a Model' if x == "" else x)

        if model_selected:
            # assign model name, ID and Framework to Deployment class object
            session_state.deployment.model_selected.name = model_selected
            session_state.deployment.model_selected.id = model_info[model_list.index(
                model_selected) - 1].ID
            session_state.deployment.model_selected.framework = model_info[model_list.index(
                model_selected) - 1].Framework

            st.write(session_state.deployment.model_selected.id,
                     session_state.deployment.model_selected.framework)
            if model_type == 'Pre-trained Models':
                # DATA_DIR
                # |_pre_trained_models/
                # | |_pt_1/
                # | | |_saved_model/
                # | |  |_saved_model.pb
                #   |_labelmap.pbtxt
                session_state.deployment.model_selected.model_path = DATA_DIR / \
                    Path(
                        [model.Model_Path for model in model_info if model.Name == model_selected][0])
                st.write(session_state.deployment.model_selected.model_path)
            elif model_type == 'Project Models':
                # project_path, training_name = session_state.deployment.model_selected.get_model_path()
                # session_state.deployment.model_selected.model_path = DATA_DIR / \
                #     project_path / get_directory_name(
                #         training_name) / 'exported_models' / get_directory_name(session_state.deployment.model_selected.name)
                session_state.deployment.model_selected.model_path = session_state.deployment.model_selected.get_model_path()
                session_state.deployment.model_selected.labelmap_path = session_state.deployment.model_selected.get_labelmap_path()
                st.write(session_state.deployment.model_selected.model_path,
                         session_state.deployment.model_selected.labelmap_path)

            if session_state.deployment.model_selected.framework == 'TensorFlow':  # SavedModel directory for TensorFlow
                session_state.deployment.model_selected.saved_model_dir = session_state.deployment.model_selected.model_path / 'saved_model'

        # ****** TODO:USER UPLOAD MODELS (KIV)*****************************************

    col1, col2, col3 = st.beta_columns(3)
    col1.write(vars(session_state.deployment))

    col2.write(vars(session_state.deployment.model_selected))
# >>>> SELECT DEPLOYMENT TYPE AND MODEL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # if model_selected:
    #     form_list = f"{deployment_type_selected:^4} ; {model_type} ; {model_selected};{model_path}"
    #     st.write(form_list)


def main():
    show()


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
