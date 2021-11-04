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
import sys
from pathlib import Path
from typing import Any, Dict
from keras_unet_collection import models
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
from training.utils import get_segmentation_model_name2func, get_training_param_from_session_state


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
        # for Anson: 4 for TFOD, 9 for img classif, 30 for segmentation
        project_id_tmp = 30
        logger.debug(f"Entering Project {project_id_tmp}")

        # session_state.append_project_flag = ProjectPermission.ViewOnly

        if "project" not in session_state:
            session_state.project = Project(project_id_tmp)
            logger.debug("Inside")
        if 'user' not in session_state:
            session_state.user = User(1)
        if 'new_training' not in session_state:
            # for Anson: 2 for TFOD, 17 for img classif, 18 for segmentation
            session_state.new_training = Training(18, session_state.project)
        # ****************************** HEADER **********************************************
        st.write(f"# {session_state.project.name}")

        project_description = session_state.project.desc if session_state.project.desc is not None else " "
        st.write(f"{project_description}")

        st.markdown("""___""")

    st.markdown(f"### Step 2: Select training configuration at sidebar.")

    deployment_type = session_state.project.deployment_type

    if deployment_type == "Semantic Segmentation with Polygons":
        train_config_col = st.sidebar.container()
        details_col = st.container()
    else:
        train_config_col, details_col = st.columns(2)

    def check_segmentation_model(training_param: Dict[str, Any] = None):
        if not training_param:
            training_param = get_training_param_from_session_state()
        model_param_dict = session_state.new_training.get_segmentation_model_params(
            training_param
        )
        with train_config_col:
            try:
                with st.spinner("Building model with the selected parameters ..."):
                    getattr(models, model_func)(**model_param_dict)
                # raise Exception("DUMMY ERROR")
            except Exception as e:
                st.error("""Error building the segmentation model with the selected 
                parameters, please try changing the parameters and try again before 
                submitting.""")
                logger.error(
                    f"Error building keras_unet_collection segmentation model: {e}")
                st.stop()
            else:
                st.success(f"""🎉 **{model_name}** Model was built 
                successfully with the selected parameters! 
                You may proceed to submit the training config.""")

    with train_config_col:
        def update_training_param():
            training_param = get_training_param_from_session_state(delete=True)
            if deployment_type == "Semantic Segmentation with Polygons":
                # continue only if the model is built successfully
                check_segmentation_model(training_param)
            # update the database and our Training instance
            session_state.new_training.update_training_param(training_param)
            session_state.new_training.has_submitted[NewTrainingPagination.TrainingConfig] = True
            logger.info(
                "Successfully submitted the selected training parameters")

            if not session_state.new_training.has_submitted[NewTrainingPagination.AugmentationConfig]:
                session_state.new_training_pagination = NewTrainingPagination.AugmentationConfig
            else:
                # go to Training page if all forms have been submitted
                session_state.new_training_pagination = NewTrainingPagination.Training
            logger.debug('New Training Pagination: '
                         f'{session_state.new_training_pagination}')
            st.experimental_rerun()

        if deployment_type != "Object Detection with Bounding Boxes":
            # NOTE: most of these params will also be used for Semantic Segmentation for Keras training
            param_dict = session_state.new_training.training_param_dict
            if param_dict:
                # details_col.write(param_dict)
                # taking the stored param from DB
                learning_rate = param_dict['learning_rate']
                optimizer = param_dict['optimizer']
                batch_size = param_dict['batch_size']
                num_epochs = param_dict['num_epochs']
                if deployment_type == "Image Classification":
                    # NOTE: not using fine_tune_all for now
                    # fine_tune_all = param_dict['fine_tune_all']
                    image_size = param_dict['image_size']
            else:
                image_size = 224
                learning_rate = 1e-4
                optimizer = "Adam"
                batch_size = 32
                num_epochs = 10
                fine_tune_all = False

            # NOTE: store them in key names starting exactly with `param_`
            #  to be able to extract them and send them over to the Trainer for training
            # e.g. param_batch_size -> batch_size at the Trainer later
            if deployment_type == "Image Classification":
                # semantic segmentation will use "input_size" in the widget later
                st.number_input(
                    "Image size", min_value=32, max_value=512,
                    value=image_size, step=1,
                    key="param_image_size",
                    help="""Image size to resize our image width and height into, e.g. 224 will
                    resize our image into size of 224 x 224. Larger image size could result in
                    better performance but most of the time it will just make the training
                    unnecessarily longer without significant improvement. Recommended to just
                    go with **224**, which is the most common input image size."""
                )
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
            if deployment_type == "Image Classification":
                # NOTE: not using fine_tune_all for now
                # st.checkbox(
                #     "Fine-tune all layers", value=fine_tune_all,
                #     key="param_fine_tune_all",
                #     help="""In most cases, our custom dataset is much smaller than the original dataset
                #     used to train the pretrained model, therefore, it is preferred to freeze
                #     (their parameters are not affected by training) all the pretrained layers,
                #     and only train the remaining layers which we will append to accommodate our
                #     custom dataset. But in some cases where we want to completely fine-tune the
                #     pretrained parameters to fit our custom dataset, we will run the training
                #     for a second time with all the pretrained model layers unfrozen.
                #     This may or may not improve the performance, depending on how much our
                #     custom dataset differs from the original dataset used for pretraining.
                #     Recommended to **start with the normal way first**, i.e. only fine-tune
                #     the last few layers (leave this unchecked)."""
                # )
                # semantic segmentation submit button will be shown later
                st.button("Submit Config", key='btn_training_config_submit',
                          on_click=update_training_param)
        else:
            # ******************************** TFOD config ********************************
            # only storing `batch_size` and `num_train_steps`
            param_dict = session_state.new_training.training_param_dict
            st.write(param_dict)
            if param_dict:
                # taking the stored param from DB
                batch_size = param_dict['batch_size']
                num_train_steps = param_dict['num_train_steps']
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

        # ****************** Model parameters for keras_unet_collection models ******************
        if deployment_type == "Semantic Segmentation with Polygons":
            # NOTE: refer to Notion for details about the model parameters
            # or refer to this Colab Notebook https://colab.research.google.com/drive/1PgI3Adcq_EixOrZm5kFsjabswxIw0c4p?usp=sharing
            param_dict = session_state.new_training.training_param_dict
            if param_dict:
                # taking the stored params from DB
                input_size = param_dict['input_size'][0]
                filter_num = param_dict['filter_num']
                # note: there is also an `n_labels` parameter initialized directly below
                filter_size = filter_num[0]
                depth = len(filter_num)
                if 'recur_num' in param_dict:
                    recur_num = param_dict['recur_num']
                stack_num_up = param_dict['stack_num_up']
                stack_num_down = param_dict['stack_num_down']
                activation = param_dict['activation']
                output_activation = param_dict['output_activation']
                batch_norm = param_dict['batch_norm']
                pool = param_dict['pool']
                unpool = param_dict['unpool']
                if 'aspp_num_down' in param_dict:
                    aspp_num_down = param_dict['aspp_num_down']
                    aspp_num_up = param_dict['aspp_num_up']
                use_hybrid_loss = param_dict['use_hybrid_loss']
            else:
                input_size = 256
                filter_size = 32
                depth = 4
                recur_num = 2
                stack_num_up = 2
                stack_num_down = 2
                activation = 'ReLU'
                output_activation = 'Softmax'
                batch_norm = True
                pool = True
                unpool = True
                aspp_num_down = 256
                aspp_num_up = 128
                use_hybrid_loss = False

            st.markdown("___")
            st.subheader("Segmentation model parameters")

            inp_choices = (128, 256, 512)
            input_size = st.select_slider(
                "Input image size", inp_choices, value=input_size, key='input_size'
            )
            # this is required as the first parameter to the segmentation model
            session_state['param_input_size'] = (input_size, input_size, 3)
            # this is required for preprocessing
            session_state['param_image_size'] = input_size

            filter_size_choices = (16, 32, 64)
            filter_size = st.select_slider(
                "Filter size for first layer", filter_size_choices,
                value=filter_size, key='filter_size', help="`filter_num` parameter"
            )
            depth_choices = (4, 5, 6)
            depth = st.select_slider(
                "Depth", depth_choices, value=depth, key='depth',
                help="""Number of filters per down/up-sampling blocks.
                e.g. If selected first layer filter size 32 with depth of 4:
                `[32, 64, 128, 256]`. i.e. 1st layer has 32 filters; 2nd: 64; 
                3rd: 128; 4th: 256. This is the conventional pattern for the 
                increasing number of filters"""
            )
            # e.g. Depth 4: [32, 64, 128, 256]
            session_state['param_filter_num'] = [
                filter_size * (2 ** i) for i in range(depth)]

            # +1 for background class
            session_state['param_n_labels'] = len(
                session_state.project.get_existing_unique_labels()) + 1

            recur_num_choices = (1, 2, 3)
            if session_state.new_training.attached_model.name == 'R2U-Net':
                st.select_slider(
                    "Number of recurrent iterations", recur_num_choices, value=recur_num,
                    key='param_recur_num', help="""Number of recurrent iterations per 
                    down- and upsampling level"""
                )

            if session_state.new_training.attached_model.name == 'ResUnet-a':
                # recommended dilation_num by author
                session_state['param_dilation_num'] = [1, 3, 5, 31]
                aspp_num_choices = (64, 128, 256)
                st.select_slider(
                    "Number of filters in ASPP up layer", aspp_num_choices, value=aspp_num_up,
                    key='param_aspp_num_up',
                    help="""Number of Atrous Spatial Pyramid Pooling (ASPP) layer 
                    filters after the last upsampling block."""
                )
                st.select_slider(
                    "Number of filters in ASPP down layer", aspp_num_choices, value=aspp_num_down,
                    key='param_aspp_num_down',
                    help="""Number of ASPP layer 
                    filters after the last downsampling block"""
                )
            else:
                stack_num_choices = (1, 2, 3)
                st.select_slider(
                    "Number of upsampling Conv layers", stack_num_choices, value=stack_num_up,
                    key='param_stack_num_up',
                    help="""Number of convolutional layers (after concatenation) 
                    per upsampling level/block"""
                )
                st.select_slider(
                    "Number of downsampling Conv layers", stack_num_choices, value=stack_num_down,
                    key='param_stack_num_down',
                    help="""Number of convolutional layers per downsampling level/block"""
                )

            activation_choices = ('ReLU', 'LeakyReLU',
                                  'PReLU', 'ELU', 'GELU', 'Snake')
            st.selectbox(
                "Activation function", activation_choices,
                index=activation_choices.index(activation),
                key='param_activation'
            )

            output_act_choices = ('Sigmoid', 'Softmax', 'Linear', 'Snake')
            # convert back from model param to the format to display to user
            output_activation = 'Linear' if output_activation is None else output_activation
            output_activation = st.selectbox(
                "Output activation function", output_act_choices,
                index=output_act_choices.index(output_activation),
                key='output_activation'
            )
            # convert to the parameter that can be accepted by the segmentation model
            output_activation = None if output_activation == 'Linear' else output_activation
            session_state['param_output_activation'] = output_activation

            st.checkbox("Batch Normalization", value=batch_norm, key='param_batch_norm',
                        help="Whether to use batch normalization layers or not")

            pool_choices = (False, True, 'Average')
            pool = 'Average' if pool == 'ave' else pool
            pool = st.selectbox(
                "Pooling layer", pool_choices,
                index=pool_choices.index(pool), key='pool',
                help="""False for strided Conv layers; True for Max Pooling;
                    'Average' for Average Pooling"""
            )
            session_state['param_pool'] = 'ave' if pool == 'Average' else pool

            unpool_choices = (False, True, 'Nearest')
            unpool = 'Nearest' if unpool == 'nearest' else unpool
            unpool = st.selectbox(
                "Unpooling layer", unpool_choices,
                index=unpool_choices.index(unpool), key='unpool',
                help="""True for Upsampling2D with bilinear interpolation.
                'Nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation. """
            )
            session_state['param_unpool'] = 'nearest' if unpool == 'Nearest' else unpool

            loss_choices = ('Hybrid Loss', 'Focal Tversky Loss')
            loss_func = 'Hybrid Loss' if use_hybrid_loss else 'Focal Tversky Loss'
            loss_func = st.radio(
                'Loss function', loss_choices, index=loss_choices.index(loss_func), key='loss_func',
                help="""Hybrid Loss is a combination of Focal Tversky Loss and Intersection
                over Union (IoU) Loss. In general, focal Tversky loss is good enough.""")
            session_state['param_use_hybrid_loss'] = (True if loss_func == 'Hybrid Loss'
                                                      else False)

            st.button("Test Build Model", key='btn_test_build_model',
                      help="""Test building a segmentation model to verify that 
                      the parameters are working""")
            st.button("Submit Config", key='btn_training_config_submit')

    if deployment_type == "Semantic Segmentation with Polygons":
        with details_col:
            model_name2_func = get_segmentation_model_name2func()
            model_name = session_state.new_training.attached_model.name
            model_func = model_name2_func[model_name]
            # show docstring
            st.subheader(f"**{model_name}** Model Docstring:")
            st.text(getattr(models, model_func).__doc__)

            # using this instead of `on_click` callbacks to show the messages
            # below the other texts
            if session_state.btn_test_build_model:
                check_segmentation_model()
            elif session_state.btn_training_config_submit:
                update_training_param()

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
