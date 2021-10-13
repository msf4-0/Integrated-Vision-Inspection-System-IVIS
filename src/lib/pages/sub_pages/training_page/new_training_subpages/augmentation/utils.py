import cv2
import os
import numpy as np
import json
import argparse

import streamlit as st
from streamlit import session_state

CONFIG_PATH = "src/lib/pages/sub_pages/training_page/new_training_subpages/augmentation/augmentations.json"


@st.experimental_memo
def get_arguments():
    """Return the values of CLI params"""
    parser = argparse.ArgumentParser()
    sample_image_path = "src/lib/pages/sub_pages/training_page/new_training_subpages/images"
    parser.add_argument("--image_folder", default=sample_image_path)
    parser.add_argument("--image_width", default=400, type=int)
    args = parser.parse_args()
    return getattr(args, "image_folder"), getattr(args, "image_width")


@st.experimental_memo
def get_images_list(path_to_folder: str) -> list:
    """Return the list of images from folder
    Args:
        path_to_folder (str): absolute or relative path to the folder with images
    """
    image_names_list = [
        x for x in os.listdir(path_to_folder) if x[-3:] in ["jpg", "peg", "png"]
    ]
    return image_names_list


@st.experimental_memo
def load_image(image_name: str, path_to_folder: str, bgr2rgb: bool = True):
    """Load the image
    Args:
        image_name (str): name of the image
        path_to_folder (str): path to the folder with image
        bgr2rgb (bool): converts BGR image to RGB if True
    """
    path_to_image = os.path.join(path_to_folder, image_name)
    image = cv2.imread(path_to_image)
    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def upload_image(bgr2rgb: bool = True):
    """Uoload the image
    Args:
        bgr2rgb (bool): converts BGR image to RGB if True
    """
    file = st.sidebar.file_uploader(
        "Upload your image (jpg, jpeg, or png)", ["jpg", "jpeg", "png"]
    )
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), 1)
    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


@st.experimental_memo
def load_augmentations_config(
    placeholder_params: dict, path_to_config: str = CONFIG_PATH
) -> dict:
    """Load the json config with params of all transforms
    Args:
        placeholder_params (dict): dict with values of placeholders
        path_to_config (str): path to the json config file
    """
    with open(path_to_config, "r") as config_file:
        augmentations = json.load(config_file)
    for name, params in augmentations.items():
        params = [fill_placeholders(param, placeholder_params)
                  for param in params]
    return augmentations


def fill_placeholders(params: dict, placeholder_params: dict) -> dict:
    """Fill the placeholder values in the config file
    Args:
        params (dict): original params dict with placeholders
        placeholder_params (dict): dict with values of placeholders
    """
    # TODO: refactor
    if "placeholder" in params:
        placeholder_dict = params["placeholder"]
        for k, v in placeholder_dict.items():
            if isinstance(v, list):
                params[k] = []
                for element in v:
                    if element in placeholder_params:
                        params[k].append(placeholder_params[element])
                    else:
                        params[k].append(element)
            else:
                if v in placeholder_params:
                    params[k] = placeholder_params[v]
                else:
                    params[k] = v
        params.pop("placeholder")
    return params


def get_params_string(param_values: dict) -> str:
    """Generate the string from the dict with parameters
    Args:
        param_values (dict): dict of "param_name" -> "param_value"
    """
    params_string = ", ".join(
        [k + "=" + str(param_values[k]) for k in param_values.keys()]
    )
    return params_string


def get_placeholder_params(image):
    return {
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "image_half_width": int(image.shape[1] / 2),
        "image_half_height": int(image.shape[0] / 2),
    }


def select_transformations(augmentations: dict, interface_type: str) -> list:
    # extract names from augmentations.json
    all_transform_names = sorted(list(augmentations.keys()))

    # extracting all the data stored in our Training instance
    existing_config = session_state.new_training.augmentation_dict
    aug = existing_config['augmentations']
    existing_transforms = list(aug.keys())

    if existing_transforms:
        # this will only have 1 transform if `interface_type` == `Simple`
        first_transform_idx = all_transform_names.index(existing_transforms[0])
    else:
        first_transform_idx = 0

    # in the Simple mode you can choose only one transform
    if interface_type == "Simple":
        transform_names = [
            st.sidebar.selectbox(
                "Select a transformation:", all_transform_names,
                index=first_transform_idx
            )
        ]
    # in the professional mode you can choose several transforms
    elif interface_type == "Professional":
        transform_names = [
            st.sidebar.selectbox(
                "Select transformation №1:", all_transform_names,
                index=first_transform_idx
            )
        ]

        while transform_names[-1] != "None":
            filtered_transform_names = all_transform_names.copy()
            for t in transform_names:
                # to avoid duplicated transforms, also make database management much easier
                filtered_transform_names.remove(t)

            current_idx = len(transform_names)
            if len(existing_transforms) > current_idx:
                selection_idx = transform_names.index(
                    existing_transforms[current_idx])
            else:
                selection_idx = 0

            transform_names.append(
                st.sidebar.selectbox(
                    f"Select transformation №{current_idx + 1}:",
                    ["None"] + filtered_transform_names,
                    index=selection_idx
                    # key=f'aug_func_{current_idx}'
                )
            )
        transform_names = transform_names[:-1]
    return transform_names


def show_random_params(data: dict, interface_type: str = "Professional"):
    """Shows random params used for transformation (from A.ReplayCompose)"""
    if interface_type == "Professional":
        st.subheader("Random params used")
        st.markdown(
            "This will be `NULL` when the transformation is not applied due to the assigned probability.")
        random_values = {}
        for applied_params in data["replay"]["transforms"]:
            random_values[
                applied_params["__class_fullname__"].split(".")[-1]
            ] = applied_params["params"]
        st.write(random_values)
