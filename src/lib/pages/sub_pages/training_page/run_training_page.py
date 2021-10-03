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
import time
from typing import Any, Dict
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state
from streamlit_tensorboard import st_tensorboard

# >>>> User-defined Modules >>>>
from core.utils.log import logger
from machine_learning.trainer import Trainer  # logger
from training.training_management import NewTrainingPagination


def index():
    logger.debug("At new_training_training_config.py")

    st.markdown("### Training Info:")

    dataset_chosen = session_state.new_training.dataset_chosen
    if len(dataset_chosen) == 1:
        dataset_chosen_str = dataset_chosen[0]
    else:
        dataset_chosen_str = []
        for idx, data in enumerate(dataset_chosen):
            dataset_chosen_str.append(f"{idx+1}. {data}")
        dataset_chosen_str = '  \n'.join(dataset_chosen_str)
    partition_ratio = session_state.new_training.partition_ratio
    st.info(f"""
    **Training Description**: {session_state.new_training.desc}  \n
    **Dataset List**: {dataset_chosen_str}  \n
    **Partition Ratio**: training : validation : test -> 
    {partition_ratio['train']} : {partition_ratio['eval']} : {partition_ratio['test']}  \n
    **Model Name**: {session_state.new_training.training_model.name}  \n
    **Model Description**: {session_state.new_training.training_model.desc}
    """)

    train_info_btn, model_info_btn, _ = st.columns([2, 1, 4])

    def back_train_info_page():
        session_state.new_training_pagination = NewTrainingPagination.InfoDataset

    with train_info_btn:
        st.button('Modify Training Info', key='btn_modify_train_info',
                  on_click=back_train_info_page)

    def back_model_page():
        session_state.new_training_pagination = NewTrainingPagination.Model

    with model_info_btn:
        st.button('Modify Model Info', key='btn_modify_model_info',
                  on_click=back_model_page)

    # ******************************** CONFIG INFO ********************************
    train_config_col, aug_config_col = st.columns([1, 1])

    with train_config_col:
        config_info = []
        for k, v in session_state.new_training.training_param_dict.items():
            # param_name = ' '.join([word.capitalize() for word in k.split('_')])
            param_name = ' '.join(k.split('_')).capitalize()
            param_val = str(v) if isinstance(v, int) else "{:.2e}".format(v)
            current_info = f'**{param_name}**: {param_val}'
            config_info.append(current_info)
        config_info = '  \n'.join(config_info)
        st.markdown('### Training Config:')
        st.info(config_info)

        def back_config_page():
            session_state.new_training_pagination = NewTrainingPagination.TrainingConfig

        st.button('Modify Training Config', key='btn_modify_config',
                  on_click=back_config_page)

        # # TODO : image classification and segmentation
        # if session_state.project.deployment_type == "Image Classification":
        #     pass
        # elif session_state.project.deployment_type == "Object Detection with Bounding Boxes":
        #     # only storing `batch_size` and `num_train_steps`
        #     # NOTE: store them in key names starting exactly with `param_`
        #     #  to be able to extract them and send them over to the Trainer for training
        #     # e.g. param_batch_size -> batch_size at the Trainer later
        #     st.number_input(
        #         "Batch size", min_value=1, max_value=128,
        #         value=session_state.training_param_dict['param_batch_size'], step=1,
        #         key="param_batch_size",
        #         help=("Update batch size based on the system's memory you"
        #               " have. Higher batch size will need a higher memory."
        #               " Recommended to start with 4. Reduce if memory warning happens."))
        #     st.number_input(
        #         "Number of training steps", min_value=100, max_value=10_000,
        #         value=session_state.training_param_dict['param_num_train_steps'],
        #         step=50, key='param_num_train_steps',
        #         help="Recommended to train for at least 2000 steps.")
        # elif session_state.project.deployment_type == "Semantic Segmentation with Polygons":
        #     pass

    with aug_config_col:
        # TODO: do this for image classification and segmentation, TFOD API does not need this
        if session_state.project.deployment_type != 'Object Detection with Bounding Boxes':
            pass

    # ******************************* Tensorboard *******************************
    # TODO: test whether this works after deployed the app
    with st.spinner("Loading TensorBoard ..."):
        st_tensorboard(logdir=session_state.new_training.training_path['tensorboard_logdir'],
                       port=6006, width=1080, scrolling=True)

    # ******************************* START TRAINING *******************************
    start_btn_col, stop_btn_col, _ = st.columns([1, 1, 5])
    warning_place = st.empty()  # for warning messages

    def start_training_callback():

        logger.info("Exporting tasks for training ...")
        session_state.project.export_tasks()

        training_param = {}
        for k, v in session_state.items():
            if k.startswith('param_'):
                # store this to keep track of current training config
                session_state.training_param_dict[k] = v
                # e.g. param_batch_size -> batch_size
                new_key = k.replace('param_', '')
                training_param[new_key] = v
        session_state.new_training.update_training_param(training_param)

        with st.spinner('Initializing training ...'):
            trainer = Trainer(session_state.project,
                              session_state.new_training)
        if trainer.deployment_type == 'Object Detection with Bounding Boxes':
            # training_param only consists of 'batch_size' and 'num_train_steps'
            trainer.run_tfod_training()
        elif trainer.deployment_type == "Image Classification":
            pass
        elif trainer.deployment_type == "Semantic Segmentation with Polygons":
            pass

    with start_btn_col:
        training_started = st.button("Start training", key='btn_start_training',
                                     on_click=start_training_callback)

    def stop_run_training():
        # BEWARE THAT THIS WILL REFRESH THE PAGE
        warning_place.warning("Training stopped!")
        time.sleep(2)
        warning_place.empty()

    with stop_btn_col:
        st.button("Stop Training", key='btn_stop_training',
                  on_click=stop_run_training)
        st.markdown('<font size="2">**NOTE**: If you click this button, '
                    'the latest progress might not be saved.</font>',
                    unsafe_allow_html=True)


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        index()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
