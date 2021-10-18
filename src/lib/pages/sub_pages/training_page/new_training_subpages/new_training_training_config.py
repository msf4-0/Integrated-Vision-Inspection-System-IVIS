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


# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[5]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

# >>>> User-defined Modules >>>>
from core.utils.log import logger  # logger
from training.training_management import NewTrainingPagination, Training
from project.project_management import Project
from user.user_management import User


def training_configuration(RELEASE=True):
    logger.debug("At new_training_training_config.py")

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
        project_id_tmp = 9
        logger.debug(f"Entering Project {project_id_tmp}")

        # session_state.append_project_flag = ProjectPermission.ViewOnly

        if "project" not in session_state:
            session_state.project = Project(project_id_tmp)
            logger.debug("Inside")
        if 'user' not in session_state:
            session_state.user = User(1)
        if 'new_training' not in session_state:
            # for Anson: 2 for TFOD, 17 for img classif
            session_state.new_training = Training(17, session_state.project)
        # ****************************** HEADER **********************************************
        st.write(f"# {session_state.project.name}")

        project_description = session_state.project.desc if session_state.project.desc is not None else " "
        st.write(f"{project_description}")

        st.markdown("""___""")

    if 'training_param_dict' not in session_state:
        session_state.training_param_dict = {}

    st.markdown(f"**Step 2: Select training configuration:** ")

    train_config_col, _ = st.columns([1, 1])

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
            session_state.new_training.has_submitted[NewTrainingPagination.TrainingConfig] = True
            session_state.new_training_pagination = NewTrainingPagination.AugmentationConfig

        if session_state.project.deployment_type == "Image Classification":
            if session_state.new_training.training_param_dict:
                # taking the stored param from DB
                learning_rate = session_state.new_training.training_param_dict['learning_rate']
                optimizer = session_state.new_training.training_param_dict['optimizer']
                batch_size = session_state.new_training.training_param_dict['batch_size']
                num_epochs = session_state.new_training.training_param_dict['num_epochs']
                fine_tune_all = session_state.new_training.training_param_dict['fine_tune_all']
            else:
                learning_rate = 1e-4
                optimizer = "Adam"
                batch_size = 32
                num_epochs = 10
                fine_tune_all = False

            # NOTE: store them in key names starting exactly with `param_`
            #  to be able to extract them and send them over to the Trainer for training
            # e.g. param_batch_size -> batch_size at the Trainer later
            with st.form(key='training_config_form'):
                lr_choices = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1)
                st.select_slider(
                    "Learning rate", lr_choices,
                    value=learning_rate,
                    # show scientific notation
                    format_func=lambda x: f"{x:.0e}",
                    key="param_learning_rate",
                    help="""This controls how much we want to update our model parameters
                    during each training step. If too low, the model will not even be able to learn,
                    or at least train very very slow. If too high, the model parameters will explode
                    and will not be able to learn too. Thus, this is a very important parameter to choose wisely.
                    Recommeded to select **1e-4** as it is the middle ground. FYI: 1e-4 = 0.0001"""
                )
                optimizer_opts = ("Adadelta", "Adagrad", "Adam", "Adamax",
                                  "Nadam", "RMSprop", "SGD")
                st.selectbox(
                    "Optimizer", optimizer_opts,
                    index=optimizer_opts.index(optimizer),
                    key="param_optimizer",
                    help="""An optimizer is responsible for updating our model parameters
                    and minimize the loss function (or error function) during training.
                    Recommeded to start with **Adam**."""
                )
                bs_choices = (4, 8, 16, 32, 64, 128)
                st.select_slider(
                    "Batch size", bs_choices,
                    value=batch_size,
                    key="param_batch_size",
                    help="""Update batch size based on the system's memory you have.
                    Higher batch size will need a higher memory. Recommended to start
                    with **32**, 64 could be fine depending on how large is the pretrained model,
                    i.e. how many parameters does it have. Reduce if memory warning happens.
                    You may choose to increase if you believe your GPU has a lot more
                    VRAM (aka memory) left"""
                )
                st.number_input(
                    "Number of epochs", min_value=3, max_value=10_000,
                    value=num_epochs, step=1,
                    key="param_num_epochs",
                    help="""Number of epochs to train your model. One epoch will go through
                    our entire dataset for exactly once. Recommended to start with **10**."""
                )
                st.checkbox(
                    "Fine-tune all layers", value=fine_tune_all,
                    key="param_fine_tune_all",
                    help="""In most cases, our custom dataset is much smaller than the original dataset
                    used to train the pretrained model, therefore, it is preferred to freeze
                    (their parameters are not affected by training) all the pretrained layers,
                    and only train the remaining layers which we will append to accommodate our
                    custom dataset. But in some cases where we want to completely fine-tune the
                    pretrained parameters to fit our custom dataset, we will run the training
                    for a second time with all the pretrained model layers unfrozen.
                    This may or may not improve the performance, depending on how much our
                    custom dataset differs from the original dataset used for pretraining.
                    Recommended to **start with the normal way first**, i.e. only fine-tune
                    the last few layers (leave this unchecked)."""
                )
                st.form_submit_button("Submit Config",
                                      on_click=update_training_param)

        elif session_state.project.deployment_type == "Object Detection with Bounding Boxes":
            # only storing `batch_size` and `num_train_steps`
            if session_state.new_training.training_param_dict:
                # taking the stored param from DB
                batch_size = session_state.new_training.training_param_dict['batch_size']
                num_train_steps = session_state.new_training.training_param_dict['num_train_steps']
            else:
                batch_size = 4
                num_train_steps = 2000

            with st.form(key='training_config_form'):
                bs_choices = (1, 2, 4, 8, 16, 32, 64, 128)
                st.select_slider(
                    "Batch size", bs_choices,
                    value=batch_size,
                    key="param_batch_size",
                    help="""Update batch size based on the system's memory you have.
                    Higher batch size will need a higher memory. Recommended to start
                    with **4**. Reduce if memory warning happens. Beware that our object
                    detection models requires a lot of memory, so do not try to simply increase
                    the batch size if you are not sure whether you have enough GPU memory."""
                )
                st.number_input(
                    "Number of training steps", min_value=100,
                    # NOTE: this max_value should be adjusted according to our server limit
                    max_value=20_000,
                    value=num_train_steps,
                    step=50, key='param_num_train_steps',
                    help="Recommended to train for at least **2000** steps."
                )
                st.form_submit_button("Submit Config",
                                      on_click=update_training_param)

        elif session_state.project.deployment_type == "Semantic Segmentation with Polygons":
            pass

    # ******************************BACK BUTTON******************************
    def to_models_page():
        session_state.new_training_pagination = NewTrainingPagination.Model

    st.sidebar.button("Back to Modify Model Info", key="training_config_back_button",
                      on_click=to_models_page)


if __name__ == "__main__":
    # DEFINE wide page layout for debugging when running this page directly
    layout = 'wide'
    st.set_page_config(page_title="Integrated Vision Inspection System",
                       page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

    if st._is_running_with_streamlit:
        training_configuration(RELEASE=False)
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
