""" 
Title: Training Page
Date: 29/9/2021 
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
import json
import sys
from pathlib import Path
from typing import Any, Dict
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state

# >>>> User-defined Modules >>>>
from core.utils.log import logger  # logger
from training.training_management import NewTrainingPagination


def index():
    logger.debug("At new_training_training_config.py")

    st.markdown(f"## Training Session: {session_state.new_training.name}")
    st.markdown(f"### Current Training config:")
    # TODO: add navigation back to modify all this info
    st.info(f"""
    **Training Description**: {session_state.new_training.desc}  \n
    **Dataset List**: {session_state.new_training.dataset_chosen}  \n
    **Partition Ratio**: {session_state.new_training.partition_ratio}  \n
    **Model Name**: {session_state.new_training.training_model.name}  \n
    **Model Description**: {session_state.new_training.training_model.desc}
    """)

    train_config_col, aug_config_col = st.columns([1, 1])

    with train_config_col:
        def update_training_param(training_param: Dict[str, Any]):
            """Callback function for form submission."""
            session_state.new_training.update_training_param(training_param)
            session_state.new_training_pagination = NewTrainingPagination.Training

        if session_state.project.deployment_type == "Image Classification":
            pass
        elif session_state.project.deployment_type == "Object Detection with Bounding Boxes":
            # only storing `batch_size` and `num_train_steps`
            st.number_input("Batch size", min_value=1, max_value=128, value=4, step=1,
                            key="batch_size",
                            help=("Update batch size based on the system's memory you"
                                  " have. Higher batch size will need a higher memory."
                                  " Recommended to start with 4. Reduce if memory warning happens."))
            st.number_input("Number of training steps", min_value=100, max_value=10_000, value=2000,
                            step=50, key='num_train_steps',
                            help="Recommended to train for at least 2000 steps.")
        elif session_state.project.deployment_type == "Semantic Segmentation with Polygons":
            pass

    with aug_config_col:
        # TODO: do this for image classification and segmentation, TFOD API does not need this
        pass


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        index()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
