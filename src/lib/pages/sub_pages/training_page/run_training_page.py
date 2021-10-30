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
import os
import shutil
import sys
from pathlib import Path
import time
from typing import Any, Dict

import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state

import tensorflow as tf

# >>>> **************** TEMP (for debugging) **************** >>>
# add the paths to be able to import them to this file
SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

# >>>> **************** TEMP **************** >>>

# >>>> User-defined Modules >>>>
from core.utils.log import logger
from machine_learning.trainer import Trainer
from machine_learning.command_utils import run_tensorboard
from machine_learning.visuals import pretty_format_param
from project.project_management import Project
from training.training_management import NewTrainingPagination, Training
from user.user_management import User


def index(RELEASE=True):
    logger.debug("Navigator: At run_training_page.py")

    # ****************** TEST ******************************
    if not RELEASE:

        # ************************TO REMOVE************************
        with st.sidebar.container():
            st.image("resources/MSF-logo.gif", use_column_width=True)
            st.title("Integrated Vision Inspection System", anchor='title')
            st.header(
                "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
            st.markdown("""___""")

        # ************************TO REMOVE************************
        # for Anson: 4 for TFOD, 9 for img classif
        project_id_tmp = 4
        logger.debug(f"Entering Project {project_id_tmp}")

        # session_state.append_project_flag = ProjectPermission.ViewOnly

        if "project" not in session_state:
            session_state.project = Project(project_id_tmp)
            logger.debug("Inside")
        if 'user' not in session_state:
            session_state.user = User(1)
        if 'new_training' not in session_state:
            # for Anson: 2 for TFOD, 17 for img classif
            session_state.new_training = Training(2, session_state.project)
        # ****************************** HEADER **********************************************
        st.write(f"# {session_state.project.name}")

        project_description = session_state.project.desc if session_state.project.desc is not None else " "
        st.write(f"{project_description}")

        st.markdown("""___""")

    def initialize_trainer():
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
            dataset_chosen_str.append(f"**{idx+1}**. {data}")
        dataset_chosen_str = '; '.join(dataset_chosen_str)
    partition_ratio = session_state.new_training.partition_ratio
    st.info(f"""
    **Training Session Name**: {session_state.new_training.name}  \n
    **Training Description**: {session_state.new_training.desc}  \n
    **Dataset List**: {dataset_chosen_str}  \n
    **Partition Ratio**: training : validation : test -> 
    {partition_ratio['train']} : {partition_ratio['eval']} : {partition_ratio['test']}  \n
    **Pretrained Model Name**: {session_state.new_training.attached_model.name}  \n
    **Model Name**: {session_state.new_training.training_model.name}  \n
    **Model Description**: {session_state.new_training.training_model.desc}
    """)

    train_info_btn, model_info_btn, _ = st.columns([2, 1, 4])

    def back_train_info_page():
        session_state.new_training_pagination = NewTrainingPagination.InfoDataset

    with train_info_btn:
        st.button('⚙️ Edit Training Info', key='btn_edit_train_info',
                  on_click=back_train_info_page)

    def back_model_page():
        session_state.new_training_pagination = NewTrainingPagination.Model

    with model_info_btn:
        st.button('⚙️ Edit Model Info', key='btn_edit_model_info',
                  on_click=back_model_page)

    if session_state.new_training.is_started:
        st.warning("✏️ **NOTE**: Only edit the model selection"
                   " if you haven't started training your model! Otherwise the information"
                   " stored in database would not be correct.")

    # ******************************** CONFIG INFO ********************************
    train_config_col, aug_config_col = st.columns([1, 1])

    with train_config_col:
        config_info = pretty_format_param(
            session_state.new_training.training_param_dict)
        st.markdown('### Training Config:')
        st.info(config_info)

        def back_config_page():
            session_state.new_training_pagination = NewTrainingPagination.TrainingConfig

        st.button('⚙️ Edit Training Config', key='btn_edit_config',
                  on_click=back_config_page)

    with aug_config_col:
        # TODO: confirm that this works for image classification and segmentation
        st.markdown('### Augmentation Config:')
        augmentation_config = session_state.new_training.augmentation_config
        if augmentation_config.exists():
            aug_config_info = pretty_format_param(
                augmentation_config.to_dict())
            st.info(aug_config_info)
        else:
            st.info("No augmentation config selected.")

        def back_aug_config_page():
            session_state.new_training_pagination = NewTrainingPagination.AugmentationConfig

        st.button('⚙️ Edit Augmentation Config', key='btn_edit_aug_config',
                  on_click=back_aug_config_page)

    if session_state.new_training.is_started:
        st.warning(
            "✏️ **NOTE**: Only edit training/augmentation config if you want to "
            "re-train or continue training!")

    st.markdown("___")

    # ******************************* START TRAINING *******************************
    train_btn_place = st.empty()
    retrain_place = st.empty()
    message_place = st.empty()  # for warning messages
    result_place = st.empty()

    def start_training_callback(is_resume=False):
        if not is_resume:
            root = session_state.new_training.training_path['ROOT']
            if root.exists():
                logger.info(f"Removing existing training directory {root}")
                shutil.rmtree(root)

            logger.info("Creating training directories ...")
            session_state.new_training.initialise_training_folder()

        def stop_run_training():
            # BEWARE that this will just refresh the page
            message_place.warning("Training stopped!")
            time.sleep(2)
            message_place.empty()

        # moved stop button into callback function to only show when training is started
        btn_stop_col, _ = st.columns([1, 1])
        with btn_stop_col:
            st.button("⛔ Stop Training", key='btn_stop_training',
                      on_click=stop_run_training)
            st.warning('✏️ **NOTE**: If you click this button, '
                       'the latest progress might not be saved.')

        logdir = session_state.new_training.training_path['tensorboard_logdir']

        if session_state.new_training.deployment_type == 'Object Detection with Bounding Boxes':
            # NOTE: the TensorBoard callback will actually create a `train` and a `validation`
            #  folders and store the logs inside this folder, so don't accidentally
            #  delete this entire folder
            logdir_folders = (logdir / 'train', logdir / 'validation')
            for p in logdir_folders:
                if p.exists():
                    # this is to avoid the problem with Tensorboard displaying
                    # overlapping graphs when continue training, probably because the
                    # TFOD 'Step' always starts from 0 even though we are continue
                    # training from an existing checkpoint
                    logger.debug("Removing existing TensorBoard logdir "
                                 f"before training: {p}")
                    shutil.rmtree(p)

        with st.spinner("Loading TensorBoard ..."):
            st.markdown("Refresh the Tensorboard by clicking the refresh "
                        "icon during training to see the progress:")
            # need to sleep for 3 seconds, otherwise the TensorBoard might accidentally
            #  load the previous checkpoints
            time.sleep(3)
            run_tensorboard(logdir)

        initialize_trainer()
        # start training, set `stdout_output` to True to print the logging outputs generated
        #  from the TFOD scripts; set to False to avoid clutterring the console outputs
        session_state.trainer.train(is_resume, stdout_output=False)

        def refresh_page():
            time.sleep(1)
            st.experimental_rerun()

        st.button("Refresh Training Page", key='btn_refresh_page',
                  help="Refresh this page to show all the results",
                  on_click=refresh_page)

    if not session_state.new_training.is_started:
        with train_btn_place.container():
            st.button("⚡ Start training", key='btn_start_training')
            if not tf.config.list_physical_devices('GPU'):
                st.warning("""WARNING: You don't have access to GPU. Training will
                take much longer time to complete without GPU.""")
            if session_state.btn_start_training:
                # must do it this way instead of using callback on button
                #  to properly show the training progress below the rendered widgets
                start_training_callback()
    else:
        with train_btn_place.container():
            st.markdown("### Training results:")
            st.button("📈 Show TensorBoard", key='btn_show_tensorboard')
            if session_state.btn_show_tensorboard:
                with st.spinner("Loading Tensorboard ..."):
                    logdir = session_state.new_training.training_path['tensorboard_logdir']
                    run_tensorboard(logdir)

            clone_col, download_col = st.columns([1, 1])

            def clone_train_session():
                session_state.new_training.clone_training_session()

                # set all the submissions as True to allow proper updates instead of inserting info into DB
                for page in session_state.new_training.has_submitted:
                    session_state.new_training.has_submitted[page] = True

                # must go back to the InfoDataset page to allow user to
                # update the temporarily created names if necessary
                session_state.new_training_pagination = NewTrainingPagination.InfoDataset

            with clone_col:
                st.button("📋 Clone the Training Session",
                          key='btn_clone_session', on_click=clone_train_session)
                st.warning('✏️ **NOTE**: This will create a new training session, '
                           'while retaining all the current configuration, '
                           'to allow you to quickly train another model for benchmarking.')

            with download_col:
                initialize_trainer()
                exist_dict = session_state.trainer.check_model_exists()

                if exist_dict['model_tarfile']:
                    def show_download_msg():
                        message_place.warning("Downloading Model ...")
                        time.sleep(2)
                        message_place.empty()
                        # remove exported directory after downloaded
                        shutil.rmtree(
                            session_state.trainer.training_path['export'])
                        # remove the tarfile after downloaded
                        os.remove(model_tarfile_path)

                    model_tarfile_path = session_state.new_training.training_path['model_tarfile']
                    with st.spinner("Preparing model to be downloaded ..."):
                        with model_tarfile_path.open(mode="rb") as fp:
                            st.download_button(
                                label="📁 Download Trained Model",
                                data=fp,
                                file_name=model_tarfile_path.name,
                                mime="application/octet-stream",
                                key="btn_download_model",
                                on_click=show_download_msg,
                            )
                        st.warning('✏️ **NOTE**: This may take awhile to download, '
                                   'depending on the file size of the trained model.')
                        if session_state.new_training.deployment_type == 'Object Detection with Bounding Boxes':
                            st.warning("""The exported object detection model will also be
                            removed after you have downloaded it. You can try checking the
                            evaluation result below after exporting, because it will load
                            from the exported SavedModel format instead of checkpoint, the
                            results could be different.""")
                # only show Export button if model checkpoint/weights file is found
                elif exist_dict['ckpt']:
                    def export_callback():
                        with message_place.container():
                            try:
                                session_state.trainer.export_model()
                                st.experimental_rerun()
                            except Exception as e:
                                st.error("""Some error has occurred when exporting,
                                    please try re-training again""")
                                logger.error(f"Error exporting model: {e}")
                                st.stop()
                    if session_state.new_training.deployment_type == 'Object Detection with Bounding Boxes':
                        label = "📁 Export TensorFlow SavedModel"
                    else:
                        label = "📁 Export TensorFlow Keras Model"
                    st.button(label, key='btn_export_model',
                              on_click=export_callback)
                    st.warning('✏️ **NOTE**: This may take awhile to export, '
                               'depending on the size of the trained model.')

        with retrain_place.container():
            retrain_col_1, resume_train_col = st.columns([1, 1])
            with retrain_col_1:
                st.button("⚡ Re-train", key='btn_retrain')
                st.warning('✏️ **NOTE**: If you re-train your model, '
                           'all the existing model data will be overwritten.')

            with resume_train_col:
                st.button("⚡ Continue training", key='btn_resume_train')
                st.warning('✏️ If you think your model needs more training, '
                           'This will continue training your model from the latest progress.')

        with result_place.container():
            metric_col_1, _ = st.columns([1, 1])
            with metric_col_1:
                metrics = pretty_format_param(
                    session_state.new_training.training_model.metrics)
                st.markdown("#### Final Metrics:")
                st.info(metrics)

                if session_state.new_training.deployment_type == 'Object Detection with Bounding Boxes':
                    with st.expander("Notions about the metrics"):
                        st.markdown(
                            "From [TFOD docs](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#training-the-model):  \n")
                        st.markdown("""Following what people have said online,
                        it seems that it is advisable to allow your model to reach a `Total loss`
                        of **at least 2** (ideally 1 and lower) if you want to achieve “fair” 
                        detection results. Obviously, lower `Total loss` is better, however very 
                        low `Total loss` should be avoided, as the model may end up overfitting 
                        the dataset, meaning that it will perform poorly when applied to images
                        outside the dataset it has been trained on. To monitor `Total loss`, as well
                        as a number of other metrics even while your model is training, have a look at 
                        the **TensorBoard** above.""")

            if session_state.new_training.training_path['models'].exists():
                initialize_trainer()
                st.markdown("___")
                st.markdown("### Evaluation results:")
                with st.spinner("Running evaluation ..."):
                    try:
                        session_state.trainer.evaluate()
                    except Exception as e:
                        if not RELEASE:
                            st.exception(e)
                        st.error("Some error has occurred. Please try "
                                 "training/exporting the model again.")
                        logger.error(f"Error evaluating: {e}")
            else:
                logger.info(f"Model {session_state.new_training.training_model_id} "
                            "is not exported yet. Skipping evaluation")

        if session_state.btn_retrain or session_state.btn_resume_train:
            # clear out the unnecessary placeholders
            train_btn_place.empty()
            result_place.empty()
            retrain_place.empty()

            is_resume = False if session_state.btn_retrain else True
            with retrain_place.container():
                start_training_callback(is_resume)


if __name__ == "__main__":
    # Set to wide page layout for debugging on this page
    layout = 'wide'
    st.set_page_config(page_title="Integrated Vision Inspection System",
                       page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)
    if st._is_running_with_streamlit:
        # This is set to False for debugging purposes
        # when running Streamlit directly from this page
        RELEASE = False

        # run this page in debugging mode
        index(RELEASE)
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
