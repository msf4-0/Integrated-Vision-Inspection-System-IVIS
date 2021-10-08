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
import os
import shutil
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
from machine_learning.trainer import Trainer, pretty_format_param, run_tensorboard  # logger
from training.training_management import NewTrainingPagination


def index():
    logger.debug("Navigator: At run_training_page.py")

    if 'trainer' not in session_state:
        # store the trainer to use for training, inference and deployment
        with st.spinner("Initializing trainer ..."):
            logger.info("Initializing trainer")
            session_state.trainer = Trainer(
                session_state.project,
                session_state.new_training)

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
        st.button('Edit Training Info', key='btn_edit_train_info',
                  on_click=back_train_info_page)

    def back_model_page():
        session_state.new_training_pagination = NewTrainingPagination.Model

    with model_info_btn:
        st.button('Edit Model Info', key='btn_edit_model_info',
                  on_click=back_model_page)

    # ******************************** CONFIG INFO ********************************
    train_config_col, aug_config_col = st.columns([1, 1])

    with train_config_col:
        config_info = pretty_format_param(
            session_state.new_training.training_param_dict)
        st.markdown('### Training Config:')
        st.info(config_info)
        if session_state.new_training.is_started:
            st.warning("**NOTE**: Do not edit model selection or training config"
                       " unless you want to re-train your model! Otherwise the information"
                       " stored in database would not be correct. ")

        def back_config_page():
            session_state.new_training_pagination = NewTrainingPagination.TrainingConfig

        st.button('Edit Training Config', key='btn_edit_config',
                  on_click=back_config_page)

    with aug_config_col:
        # TODO: do this for image classification and segmentation, TFOD API does not need this
        if session_state.project.deployment_type != 'Object Detection with Bounding Boxes':
            pass

    # ******************************* START TRAINING *******************************
    train_btn_place = st.empty()
    retrain_place = st.empty()
    warning_place = st.empty()  # for warning messages
    result_place = st.empty()

    def start_training_callback():
        root = session_state.new_training.training_path['ROOT']
        if root.exists():
            logger.info(f"Removing existing training directory {root}")
            shutil.rmtree(root)

        logger.info("Creating training directories ...")
        session_state.new_training.initialise_training_folder()

        def stop_run_training():
            # BEWARE that this will just refresh the page
            warning_place.warning("Training stopped!")
            time.sleep(2)
            warning_place.empty()

        # moved stop button into callback function to only show when training is started
        col, _ = st.columns([1, 1])
        with col:
            st.button("Stop Training", key='btn_stop_training',
                      on_click=stop_run_training)
            st.warning('**NOTE**: If you click this button, '
                       'the latest progress might not be saved.')

        with st.spinner("Loading TensorBoard ..."):
            st.markdown("Refresh the Tensorboard by clicking the refresh "
                        "icon during training to see the progress:")
            logdir = session_state.new_training.training_path['tensorboard_logdir']
            # moved tensorboard into callback function to make sure it stays visible
            run_tensorboard(logdir)

        with st.spinner("Exporting tasks for training ..."):
            session_state.project.export_tasks()

        # start training
        session_state.trainer.train()

        # rerun to remove all these progress and refresh the page to show results
        st.experimental_rerun()

    if not session_state.new_training.is_started:
        with train_btn_place.container():
            st.button("Start training", key='btn_start_training')
            if session_state.btn_start_training:
                # must do it this way instead of using callback on button
                #  to properly show the training progress below the rendered widgets
                start_training_callback()
    else:
        # ? Maybe add an option for continue training
        with train_btn_place.container():
            st.markdown("___")
            st.markdown("### Training results:")
            st.button("Show TensorBoard", key='btn_show_tensorboard')
            if session_state.btn_show_tensorboard:
                with st.spinner("Loading Tensorboard ..."):
                    logdir = session_state.new_training.training_path['tensorboard_logdir']
                    run_tensorboard(logdir)

        with retrain_place.container():
            col, _ = st.columns([1, 1])
            with col:
                st.button("Re-train", key='btn_retrain')
                st.warning('**NOTE**: If you re-train your model, '
                           'all the existing model data will be overwritten.')

            # TODO: fix the weird shadow of other widgets after clicked this button
            if session_state.btn_retrain:
                # clear out the results container
                train_btn_place.empty()
                result_place.empty()
                start_training_callback()

        with result_place.container():
            col, _ = st.columns([1, 1])
            with col:
                metrics = pretty_format_param(
                    session_state.new_training.training_model.metrics)
                st.markdown("#### Final Metrics:")
                st.info(metrics)

            if session_state.new_training.training_path['model_tarfile'].exists():
                st.markdown("### Evaluation results:")
                # show evaluation results
                with st.spinner("Running evaluation ..."):
                    session_state.trainer.evaluate()
            else:
                logger.info(f"Model {session_state.new_training.training_model_id} "
                            "is not exported yet. Skipping evaluation")

        # if session_state.btn_retrain:
        #     # clear out the results container
        #     train_btn_place.empty()
        #     result_place.empty()
        #     start_training_callback()


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        index()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
