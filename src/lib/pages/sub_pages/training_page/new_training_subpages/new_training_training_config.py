""" 
Title: Training Parameters Configuration (New Training Configuration)
Date: 11/9/2021 
Author: Chu Zhen Hao
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
import json
import sys
from pathlib import Path
from typing import Any, Dict
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state


# DEFINE Web APP page configuration
layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[5]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib


# >>>> User-defined Modules >>>>
from core.utils.log import logger  # logger
from training.training_management import NewTrainingPagination


def training_configuration():
    logger.debug("At new_training_training_config.py")

    if 'training_param_dict' not in session_state:
        session_state.training_param_dict = {}

    st.markdown(f"**Step 2: Select training configuration:** ")

    train_config_col, aug_config_col = st.columns([1, 1])

    with train_config_col:
        def update_training_param():
            training_param = {}
            for k, v in session_state.items():
                if k.startswith('param_'):
                    # store this to keep track of current training config startswith 'param_'
                    session_state.training_param_dict[k] = v
                    # e.g. param_batch_size -> batch_size
                    new_key = k.replace('param_', '')
                    training_param[new_key] = v
            # update the database and our Training instance
            session_state.new_training.update_training_param(training_param)
            session_state.new_training_pagination = NewTrainingPagination.Training

        if session_state.project.deployment_type == "Image Classification":
            pass
        elif session_state.project.deployment_type == "Object Detection with Bounding Boxes":
            # only storing `batch_size` and `num_train_steps`
            # NOTE: store them in key names starting exactly with `param_`
            #  refer run_training_page for more info
            with st.form(key='training_config_form'):
                st.number_input(
                    "Batch size", min_value=1, max_value=128, value=4, step=1,
                    key="param_batch_size",
                    help=("Update batch size based on the system's memory you"
                          " have. Higher batch size will need a higher memory."
                          " Recommended to start with 4. Reduce if memory warning happens.")
                )
                st.number_input(
                    "Number of training steps", min_value=100, max_value=10_000, value=2000,
                    step=50, key='param_num_train_steps',
                    help="Recommended to train for at least 2000 steps."
                )
                st.form_submit_button("Submit", on_click=update_training_param,
                                      #   kwargs={"param_batch_size": param_batch_size,
                                      #           "param_num_train_steps": param_num_train_steps}
                                      )
        elif session_state.project.deployment_type == "Semantic Segmentation with Polygons":
            pass

    with aug_config_col:
        # TODO: do this for image classification and segmentation, TFOD API does not need this
        pass

    # ******************************BACK BUTTON******************************
    if session_state.new_training_pagination == NewTrainingPagination.TrainingConfig:
        def to_models_page():
            session_state.new_training_pagination = NewTrainingPagination.Model

        st.button("Modify Model Info", key="training_config_back_button",
                  on_click=to_models_page)


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        training_configuration()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
