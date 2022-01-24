"""
Title: Labelmap Generator
Date: 8/9/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
Description: Text area widget to generate labelmap for Computer Vision

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

from enum import IntEnum
import sys
from pathlib import Path
from typing import Union
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state


# DEFINE Web APP page configuration
# layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib


# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from core.utils.log import logger
from data_manager.database_manager import init_connection
from training.labelmap_management import Labels
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration <<<<
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])
# <<<< Variable Declaration <<<<


def labelmap_generator(framework: Union[str, IntEnum] = None, deployment_type: Union[str, IntEnum] = None) -> str:
    """Text Area widget to generate Labelmaps for Computer Vision

    Args:
        framework (Union[str, IntEnum]): Deep Learning Framework
        deployment_type (Union[str, IntEnum]): Deployment Type of model

    Returns:
        str: Labelmap string encoded in 'utf-8'
    """
    chdir_root()  # change to root directory

    st.checkbox(label='Generate Labelmap', key='generate_labelmap_checkbox')
    labelmap_string = ''
    if session_state.generate_labelmap_checkbox:
        st.info("Enter your labels separated by a comma (,)")
        st.warning("""Note that the **order of the labels** is very important. 
            It should follow the order that was used to train your model.""")
        if deployment_type == 'Semantic Segmentation with Polygons':
            st.warning(
                "For semantic segmentation, a background class should also be available.")
        st.text_area(
            label='Labels',
            value='Enter your labels separated by a comma (,)',
            key='labels_text_input')

        logger.debug(f"{framework = }, {deployment_type = }")

        if framework and deployment_type:

            if session_state.labels_text_input:
                labels_list = Labels.generate_list_of_labels(
                    session_state.labels_text_input)
                logger.debug(f"{labels_list = }")
                labelmap_string = Labels.generate_labelmap_string(labels_list=labels_list,
                                                                  framework=framework,
                                                                  deployment_type=deployment_type)

        else:
            st.warning(
                f"Please select Framework and Deployment Type to produce the correct labelmap format")
            if session_state.labels_text_input:
                labelmap_string = Labels.generate_list_of_labels(
                    session_state.labels_text_input)

        with st.expander(label='Labelmap', expanded=True):

            st.code(body=labelmap_string, language='json')

    return session_state.generate_labelmap_checkbox, labelmap_string


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        labelmap_generator(framework='TensorFlow',
                           deployment_type='Object Detection with Bounding Boxes')
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
