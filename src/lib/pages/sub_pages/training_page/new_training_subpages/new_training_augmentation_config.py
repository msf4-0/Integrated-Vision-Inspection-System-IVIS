""" Copyright (C) 2021 Selangor Human Resource Development Centre

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
import sys
from pathlib import Path
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state
import albumentations as A

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[5]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib


# DEFINE wide page layout for debugging on this page
layout = 'wide'
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)


# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from core.utils.log import logger
from data_manager.database_manager import init_connection
from training.training_management import NewTrainingPagination, Training
from project.project_management import Project
from user.user_management import User

# augmentation config from https://github.com/IliaLarchenko/albumentations-demo/blob/master/src/app.py
from augmentation.utils import (
    load_augmentations_config,
    get_arguments,
    get_placeholder_params,
    select_transformations,
    show_random_params,
)
from augmentation.visuals import (
    select_image,
    show_docstring,
    get_transormations_params,
)

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration <<<<
conn = init_connection(**st.secrets["postgres"])
# <<<< Variable Declaration <<<<


def augmentation_configuration(RELEASE=True):
    logger.debug("NAVIGATOR: At new_training_augmentation_config.py")

    if not RELEASE:
        # ************************TO REMOVE************************
        with st.sidebar.container():
            st.image("resources/MSF-logo.gif", use_column_width=True)
            st.title("Integrated Vision Inspection System", anchor='title')
            st.header(
                "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
            st.markdown("""___""")

        # ************************TO REMOVE************************
        project_id_tmp = 4
        logger.debug(f"Entering Project {project_id_tmp}")

        # session_state.append_project_flag = ProjectPermission.ViewOnly

        if "project" not in session_state:
            session_state.project = Project(project_id_tmp)
            logger.debug("Inside")
        if 'user' not in session_state:
            session_state.user = User(1)
        if 'new_training' not in session_state:
            session_state.new_training = Training(2, session_state.project)
        # ****************************** HEADER **********************************************
        st.write(f"# {session_state.project.name}")

        project_description = session_state.project.desc if session_state.project.desc is not None else " "
        st.write(f"{project_description}")

        st.markdown("""___""")

    # ************************ Session state ************************

    if 'augment_config' not in session_state:
        # this is to store all the config, to send to callback on submit to update database
        # You can refer to augmentation/sample_augment_config.json for a sample output
        session_state.augment_config = {
            'interface_type': 'Simple', 'augmentations': {}}

    # - This is for None or empty dict; also to create a nested `augmentations` Dict inside
    if not session_state.new_training.augmentation_dict:
        session_state.new_training.augmentation_dict = {}
        session_state.new_training.augmentation_dict['augmentations'] = {
            'interface_type': 'Simple',
            "augmentations": {}
        }
    elif not session_state.new_training.augmentation_dict['augmentations']:
        session_state.new_training.augmentation_dict['augmentations'] = {}

    # ******************************BACK BUTTON******************************
    def to_training_config_page():
        session_state.new_training_pagination = NewTrainingPagination.TrainingConfig

    st.sidebar.button("Back to Modify Training Config", key="augment_config_back_button",
                      on_click=to_training_config_page)

    # ************************ Config column ************************

    # get CLI params: the path to images and image width
    path_to_images, width_original = get_arguments()

    if not os.path.isdir(path_to_images):
        st.title("There is no directory: " + path_to_images)
        st.stop()

    # select interface type
    options = ("Simple", "Professional")
    curr_idx = options.index(
        session_state.new_training.augmentation_dict['interface_type'])
    interface_type = st.sidebar.radio(
        "Select the interface mode",
        options,
        index=curr_idx,
        key='aug_interface_type'
    )
    # update this to store in DB later
    session_state['augment_config']['interface_type'] = interface_type

    # select image
    status, image = select_image(path_to_images, interface_type)
    if status == 1:
        st.title("Can't load image")
    if status == 2:
        st.title("Please, upload the image")
    else:
        # image was loaded successfully
        placeholder_params = get_placeholder_params(image)

        # load the config from augmentations.json
        augmentations = load_augmentations_config(placeholder_params)

        # get the list of transformations names
        transform_names = select_transformations(
            augmentations, interface_type)

        # reset augment_config to avoid storing all unwanted previous selections
        session_state.augment_config = {
            'interface_type': session_state.aug_interface_type,
            'augmentations': {}
        }

        # get parameters for each transform
        transforms = get_transormations_params(
            transform_names, augmentations)

        try:
            # apply the transformation to the image
            data = A.ReplayCompose(transforms)(image=image)
            error = 0
        except ValueError:
            error = 1
            st.title(
                "The error has occurred. Most probably you have passed wrong set of parameters. \
            Check transforms that change the shape of image."
            )

        # ********************** Details column **********************

        # proceed only if everything is ok
        if error == 0:
            augmented_image = data["image"]
            # show title
            st.markdown("## Demo of Image Augmentation")
            st.markdown(
                f"**Step 3: Select augmentation configuration at sidebar "
                "and see the changes in the transformed image.**")
            st.markdown(
                "The `Professional` mode allows more than one transformation, "
                "and also the option to upload an image of your choice.")

            # show the images
            width_transformed = int(
                width_original / image.shape[1] * augmented_image.shape[1]
            )

            st.image(image, caption="Original image", width=width_original)
            st.image(
                augmented_image,
                caption="Transformed image",
                width=width_transformed,
            )

            # comment about refreshing
            st.markdown(
                "*Press 'R' to refresh to try for different random results*")

            # random values used to get transformations
            show_random_params(data, interface_type)

            # print additional info
            for transform in transforms:
                show_docstring(transform)
                st.code(str(transform))

    # ******************************SUBMIT BUTTON******************************

    def update_augment_config():
        # update the database and our Training instance
        session_state.new_training.update_augment_config(
            session_state.augment_config)
        session_state.new_training.has_submitted[NewTrainingPagination.AugmentationConfig] = True
        session_state.new_training_pagination = NewTrainingPagination.Training

    st.sidebar.button("Submit Augmentation Config", key="augment_config_submit_button",
                      on_click=update_augment_config)

    st.markdown("___")
    st.markdown("**Acknowledgement**: Huge thanks to [albumentations](https://github.com/IliaLarchenko/albumentations-demo) "
                "for the amazing data augmentation library and also the [Streamlit demo](https://albumentations-demo.herokuapp.com/) as reference.")


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        augmentation_configuration(RELEASE=False)
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
