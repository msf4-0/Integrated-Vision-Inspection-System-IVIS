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
import time
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state
import albumentations as A

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[5]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from core.utils.log import logger
from data_manager.database_manager import init_connection
from training.training_management import AugmentationConfig, NewTrainingPagination, Training
from project.project_management import Project
from user.user_management import User
from machine_learning.utils import get_bbox_label_info, xml_to_df
from machine_learning.visuals import draw_gt_bbox

# augmentation config from https://github.com/IliaLarchenko/albumentations-demo/blob/master/src/app.py
from .augmentation.utils import (
    load_augmentations_config,
    get_placeholder_params,
    select_transformations,
    show_random_params,
)
from .augmentation.visuals import (
    select_image,
    show_docstring,
    get_transormations_params,
    show_bbox_params_selection,
    show_train_size_selection
)


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

    if 'augmentation_config' not in session_state:
        # this is to store all the config, to send to callback on submit to update database
        # You can refer to augmentation/sample_augment_config.json for a sample output
        session_state.augmentation_config = AugmentationConfig()

    # # - This is for None or empty dict; also to create a nested `augmentations` Dict inside
    # if not session_state.new_training.has_augmentation():
    #     session_state.new_training.augmentation_config = AugmentationConfig()

    # ******************************BACK BUTTON******************************
    def to_training_config_page():
        session_state.new_training_pagination = NewTrainingPagination.TrainingConfig

    st.sidebar.button("Back to Modify Training Config", key="btn_back_train_config",
                      on_click=to_training_config_page)

    def skip_augmentation():
        session_state.augmentation_config.reset()
        # update the database and our Training instance
        session_state.new_training.update_augment_config(
            session_state.augmentation_config)
        # set the current page as submitted and move to next page
        session_state.new_training.has_submitted[NewTrainingPagination.AugmentationConfig] = True
        session_state.new_training_pagination = NewTrainingPagination.Training

    st.sidebar.button("Skip augmentation", key="btn_augment_config_skip",
                      on_click=skip_augmentation)
    st.sidebar.info("""NOTE: You can skip augmentation if you deem it's not necessary. 
    It is completely optional, although image augmentation is beneficial in most cases.""")

    # ************************ Config column ************************

    # get project exported dataset folder
    exported_dataset_dir = session_state.project.get_export_path()
    if not exported_dataset_dir.exists():
        # export the dataset with the correct structure if not done yet
        session_state.project.export_tasks()
    if session_state.new_training.deployment_type == 'Image Classification':
        image_folder = exported_dataset_dir
    elif session_state.new_training.deployment_type == 'Object Detection with Bounding Boxes':
        image_folder = exported_dataset_dir / "images"
        bbox_label_folder = exported_dataset_dir / "Annotations"
    elif session_state.new_training.deployment_type == 'Semantic Segmentation with Polygons':
        image_folder = exported_dataset_dir / "images"
        coco_json_path = exported_dataset_dir / "result.json"

    if not os.path.isdir(image_folder):
        st.title("There is no directory: " + image_folder)
        st.stop()

    # reset augmentation_config to avoid storing all unwanted previous selections
    # when the user changed the transformations
    session_state.augmentation_config.reset()

    # select the number of images to generate from the augmentation, ONLY needed for TFOD
    if session_state.new_training.deployment_type == 'Object Detection with Bounding Boxes':
        ori_train_size, aug_train_size = show_train_size_selection()
        diff = aug_train_size - ori_train_size
        pct_change = round(diff / ori_train_size * 100, 1)
        st.sidebar.info(f"Relative increase: **{diff} ({pct_change}%)**  \n"
                        f"Original training size: **{ori_train_size}**")

    # select interface type
    options = ("Simple", "Professional")
    curr_idx = options.index(
        session_state.new_training.augmentation_config.interface_type)
    interface_type = st.sidebar.radio(
        "Select the interface mode",
        options,
        index=curr_idx,
        key='aug_interface_type'
    )
    # update this to store in DB later
    session_state.augmentation_config.interface_type = interface_type

    # select image
    status, image, image_name = select_image(image_folder, interface_type, 50)
    if status == 1:
        st.title("Can't load image")
    if status == 2:
        st.title("Please, upload the image")
    else:
        # image was loaded successfully
        placeholder_params = get_placeholder_params(image)

        # load the config from augmentations.json
        augmentations = load_augmentations_config(placeholder_params)

        # show the widgets and get the list of transformations names
        transform_names = select_transformations(
            augmentations, interface_type)
        st.sidebar.markdown("___")

        if session_state.new_training.deployment_type == 'Object Detection with Bounding Boxes':
            min_area, min_visibility = show_bbox_params_selection()
            st.sidebar.markdown("___")

            if image_name != "Upload my image":
                xml_df = xml_to_df(bbox_label_folder)
                class_names, bboxes = get_bbox_label_info(xml_df, image_name)

        # show the widgets and get parameters for each transform
        transforms = get_transormations_params(
            transform_names, augmentations)

        try:
            if image_name != "Upload my image":
                if session_state.new_training.deployment_type == 'Object Detection with Bounding Boxes':
                    # apply the transformation to the image with consideration of bboxes
                    data = A.ReplayCompose(
                        transforms,
                        bbox_params=A.BboxParams(
                            format='pascal_voc',
                            min_area=min_area,
                            min_visibility=min_visibility,
                            label_fields=['class_names']
                        )
                    )(image=image, bboxes=bboxes, class_names=class_names)
            # TODO for mask augmentation
            else:
                # apply the transformation to the image
                data = A.ReplayCompose(transforms)(image=image)
            error = 0
        except ValueError:
            error = 1
            st.title(
                "The error has occurred. Most probably you have passed wrong set of parameters. \
            Check transforms that change the shape of image."
            )
        except NotImplementedError as e:
            error = 1
            st.error(f"""Transformation of **{str(e).split()[-1]}** is not available for 
            the current computer vision task: **{session_state.new_training.deployment_type}**.
            Please try another transformation.""")
            st.stop()

        # ********************** Details column **********************

        # proceed only if everything is ok
        if error == 0:
            augmented_image = data["image"]
            # show title
            st.markdown(
                f"### Step 3: Select augmentation configuration at sidebar "
                "and see the changes in the transformed image.")
            st.markdown(
                """The `Professional` mode allows more than one transformation, 
                and also the option to upload an image of your choice.""")
            st.info("""
                ✏️ NOTE: Image augmentation may be very useful to increase the number of 
                images in our training set, but it could hurt your deep learning algorithm 
                if your label has changed after the transformation, e.g. a 
                horizontally-flipped digit would not be representative of a real scenario 
                as it is not a correct digit anymore. Therefore, choose your transformation
                techniques wisely. Recommended to start with `RandomBrightness`,
                `RandomBrightnessContrast` or similar ones that will not tremendously
                alter the appearance of the relevant objects in the image.""")
            st.markdown("#### Demo of image augmentation result")

            # Set this image width for the size of our image to display on Streamlit
            # width_original = 400
            # # show the images
            # width_transformed = int(
            #     width_original / image.shape[1] * augmented_image.shape[1]
            # )

            if image_name != "Upload my image":
                if session_state.new_training.deployment_type == 'Object Detection with Bounding Boxes':
                    image = draw_gt_bbox(
                        image, bboxes, class_names=class_names)
                    augmented_image = draw_gt_bbox(augmented_image, data['bboxes'],
                                                   class_names=data['class_names'])
            # TODO: add for mask augmentation

            true_img_col, aug_img_col = st.columns(2)
            with true_img_col:
                st.image(image, caption="Original image",
                         use_column_width=True)
            with aug_img_col:
                st.image(
                    augmented_image,
                    caption="Transformed image",
                    use_column_width=True,
                )

            # comment about refreshing
            st.markdown(
                "*Press 'R' to refresh to try for different random results*")

            st.markdown("___")

            if interface_type == "Professional":
                random_param_col, docstring_col = st.columns([1, 2])
                with random_param_col:
                    # random values used to get transformations
                    show_random_params(data, interface_type)
            else:
                docstring_col = st.container()

            with docstring_col:
                # print additional info
                for transform in transforms:
                    show_docstring(transform)
                    st.code(str(transform))

    # ******************************SUBMIT BUTTON******************************

    def update_augment_config():
        # update the database and our Training instance
        session_state.new_training.update_augment_config(
            session_state.augmentation_config)
        session_state.new_training.has_submitted[NewTrainingPagination.AugmentationConfig] = True
        session_state.new_training_pagination = NewTrainingPagination.Training

    st.sidebar.button("Submit Augmentation Config", key="btn_augment_config_submit",
                      on_click=update_augment_config)

    st.markdown("___")
    st.markdown("**Acknowledgement**: Huge thanks to [albumentations](https://github.com/IliaLarchenko/albumentations-demo) "
                "for the amazing data augmentation library and also the [Streamlit demo](https://albumentations-demo.herokuapp.com/) as reference.")

    st.write("session_state.augmentation_config")
    st.write(session_state.augmentation_config)


if __name__ == "__main__":
    # DEFINE wide page layout for debugging on this page
    layout = 'wide'
    st.set_page_config(page_title="Integrated Vision Inspection System",
                       page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)
    conn = init_connection(**st.secrets["postgres"])

    if st._is_running_with_streamlit:
        augmentation_configuration(RELEASE=False)
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
