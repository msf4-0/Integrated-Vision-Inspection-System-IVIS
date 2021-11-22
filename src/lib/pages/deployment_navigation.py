"""
Title: Deployment navigation
Date: 16/11/2021
Author: Anson Tan Chen Tung
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)

Copyright (C) 2021 Selangor Human Resource Development Centre

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. 

Copyright (C) 2021 Selangor Human Resource Development Centre
SPDX-License-Identifier: Apache-2.0
========================================================================================
"""
import sys
from pathlib import Path
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state


# DEFINE Web APP page configuration for debugging on this page
layout = 'wide'
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

from path_desc import chdir_root
from core.utils.log import logger
from project.project_management import Project
from training.model_management import NewModel
from deployment.deployment_management import DeploymentPagination
from deployment.utils import reset_camera
from pages.sub_pages.models_page.models_subpages.user_model_upload import user_model_upload_page
from pages.sub_pages.deployment_page import model_selection, deployment_page


def index(RELEASE=True):
    logger.debug("At deployment navigation")
    # ****************** TEST ******************************
    if not RELEASE:
        # ************************TO REMOVE************************
        with st.sidebar.container():
            st.image("resources/MSF-logo.gif", use_column_width=True)
            st.title("Integrated Vision Inspection System", anchor='title')
            st.header(
                "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
            st.markdown("""___""")
        # for Anson: 4 for TFOD, 9 for img classif, 30 for segmentation
        # uploaded pet segmentation: 96
        # uploaded face detection: 111
        project_id_tmp = 111
        logger.debug(f"Entering Project {project_id_tmp}")
        if "project" not in session_state:
            session_state.project = Project(project_id_tmp)

        # ****************************** HEADER **********************************************
        st.write(f"# Project: {session_state.project.name}")

        project_description = session_state.project.desc if session_state.project.desc is not None else " "
        st.write(f"{project_description}")

        st.markdown("___")

    # ************************ Deployment PAGINATION *************************
    deployment_pagination2func = {
        DeploymentPagination.Models: model_selection.index,
        DeploymentPagination.UploadModel: user_model_upload_page,
        DeploymentPagination.Deployment: deployment_page.index
    }

    if 'deployment_pagination' not in session_state:
        session_state.deployment_pagination = DeploymentPagination.Models

    # >>>> Pagination RADIO >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    deployment_page_options = ("Model Selection", "Upload Model", "Deployment")

    def deployment_page_navigator():
        navigation_selected = session_state.deployment_page_navigator_radio
        navigation_selected_idx = deployment_page_options.index(
            navigation_selected)
        session_state.deployment_pagination = navigation_selected_idx

        # reset the camera to give back access to user
        reset_camera()
        if navigation_selected == "Model Selection":
            pass
        elif navigation_selected == "Upload Model":
            NewModel.reset_model_upload_page()
        elif navigation_selected == "Deployment":
            pass

    with st.sidebar.expander(session_state.project.name, expanded=True):
        st.radio("Deployment Navigation", options=deployment_page_options,
                 index=session_state.deployment_pagination,
                 on_change=deployment_page_navigator,
                 key="deployment_page_navigator_radio")
    st.sidebar.markdown("___")

    # >>>> MAIN FUNCTION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.debug(f"{session_state.deployment_pagination = }")
    deployment_pagination2func[session_state.deployment_pagination]()


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        # False for debugging
        index(RELEASE=False)
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
