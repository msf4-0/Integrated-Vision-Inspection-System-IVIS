import sys
from typing import Tuple
import cv2
from pathlib import Path
import streamlit as st
from streamlit import session_state

import albumentations as A

from .control import param2func
from .utils import get_images_list, load_image, upload_image

SRC = Path(__file__).resolve().parents[5]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

# >>>> User-defined Modules >>>>
from core.utils.log import logger

# set this to False to stop showing debugging values on sidebar
DEBUG = False


def select_image(path_to_images: str, interface_type: str = "Simple", n_images: int = 10):
    """ Show interface to choose the image, and load it

    Args:
        path_to_images (dict): path ot folder with images
        interface_type (dict): mode of the interface used
        n_images (int): maximum number of images to display as options

    Returns:
        (status, image)
        status (int):
            0 - if everything is ok
            1 - if there is error during loading of image file
            2 - if user hasn't uploaded photo yet
    """
    image_names_list, image_paths = get_images_list(path_to_images, n_images)
    if len(image_names_list) < 1:
        return 1, 0, None
    else:
        if interface_type == "Professional":
            image_name = st.sidebar.selectbox(
                "Select an image:", image_names_list + ["Upload my image"]
            )
        else:
            image_name = st.sidebar.selectbox(
                "Select an image:", image_names_list)

        if image_name != "Upload my image":
            try:
                idx = image_names_list.index(image_name)
                image = load_image(image_paths[idx])
                return 0, image, image_name
            except cv2.error:
                return 1, 0, image_name
        else:
            # all of these will return `image_name` of "Upload my image"
            try:
                image = upload_image()
                return 0, image, image_name
            except cv2.error:
                return 1, 0, image_name
            except AttributeError:
                return 2, 0, image_name


def show_transform_control(transform_params: dict,
                           n_for_hash: int,
                           existing_params: dict = None) -> dict:
    st.sidebar.subheader("probability")
    if existing_params:
        current_p = existing_params['p']
    else:
        existing_params = {}
        current_p = 1.0
    p = st.sidebar.slider("", 0., 1., current_p, key=f'aug_p_{n_for_hash}',
                          help=('Probability of applying the transformation.  \n'
                                'e.g. 0.50 = 50% chance of applying the transformation'))
    param_values = {'p': p}

    if len(transform_params) == 0:
        # Added slider widget for probability
        # st.sidebar.text("Transform has no parameters")
        pass
    else:
        for param in transform_params:
            control_type = param["type"]

            if isinstance(param['param_name'], list):
                param_name_to_check = param['param_name'][0]
            else:
                param_name_to_check = param['param_name']

            if param_name_to_check in existing_params:
                # use the existing params stored in DB if available
                if control_type in ('num_interval', 'checkbox'):
                    default_key = 'defaults'
                elif control_type == 'radio':
                    default_key = 'default_str'
                else:
                    # elif control_type == ('several_nums', 'rgb', 'min_max'):
                    default_key = 'defaults_list'
                param[default_key] = existing_params[param['param_name']]

            # this is where we get the Streamlit function and show the widget
            control_function = param2func[control_type]
            if isinstance(param["param_name"], list):
                returned_values = control_function(
                    **param, n_for_hash=n_for_hash)
                for name, value in zip(param["param_name"], returned_values):
                    param_values[name] = value
            else:
                # if param['param_name'] in existing_params:
                #     param['defaults'] = existing_params[param['param_name']]

                param_values[param["param_name"]] = control_function(
                    **param, n_for_hash=n_for_hash
                )
    return param_values


def show_credentials():
    st.markdown("* * *")
    st.subheader("Credentials:")
    st.markdown(
        (
            "Source: [github.com/IliaLarchenko/albumentations-demo]"
            "(https://github.com/IliaLarchenko/albumentations-demo)"
        )
    )
    st.markdown(
        (
            "Albumentations library: [github.com/albumentations-team/albumentations]"
            "(https://github.com/albumentations-team/albumentations)"
        )
    )
    st.markdown(
        (
            "Image Source: [pexels.com/royalty-free-images]"
            "(https://pexels.com/royalty-free-images/)"
        )
    )


def get_transformations_params(transform_names: list, augmentations: dict) -> list:
    existing_aug = session_state.new_training.augmentation_config.augmentations

    transforms = []
    for i, transform_name in enumerate(transform_names):
        if existing_aug and (transform_name in existing_aug):
            existing_params = existing_aug[transform_name]
        else:
            existing_params = None

        # select the params values
        st.sidebar.subheader("Params of the " + transform_name)
        param_values = show_transform_control(
            augmentations[transform_name], i, existing_params)
        if DEBUG:
            logger.debug(f"{param_values = }")
            st.sidebar.write(param_values)
        # store the augmentation's transform name and param_values here
        session_state.augmentation_config.augmentations[transform_name] = param_values

        transforms.append(getattr(A, transform_name)(**param_values))
    return transforms


def show_train_size_selection() -> Tuple[int, int]:
    if session_state.new_training.partition_size['train'] == 0:
        session_state.new_training.calc_dataset_partition_size(
            session_state.new_training.dataset_chosen,
            session_state.project.dataset_dict
        )
    train_size = session_state.new_training.partition_size['train']
    aug_train_size = st.sidebar.number_input(
        "Select the number of images to generate",
        min_value=train_size, max_value=train_size * 5,
        value=train_size * 2, step=1, key='aug_train_size',
        help="""This is the total number of images that will be augmented and use
        for training (`train_size`). This is only needed for object detection task because we are
        generating the images before training, rather than augmenting them on the fly.
        NOTE: only allows a maximum relative increase of up to 400%"""
    )
    # store it in our config to store in DB
    session_state.augmentation_config.train_size = aug_train_size
    return train_size, aug_train_size


def show_bbox_params_selection() -> Tuple[int, float]:
    st.sidebar.subheader("Bounding box parameters")
    existing_config = session_state.new_training.augmentation_config
    if existing_config.min_area:
        min_area = existing_config.min_area
    else:
        min_area = 200  # default suggested value
    if existing_config.min_visibility:
        min_visibility = existing_config.min_visibility
    else:
        min_visibility = 0.1  # default suggested value

    min_area = st.sidebar.slider(
        "Minimum visible area", 1, 20_000, min_area, 1,
        key='bbox_min_area',
        help="""**Minimum visible area** is a value in pixels. If the area 
            of a bounding box after augmentation becomes smaller than **Minimum 
            visible area**, the transform will drop that box. So the returned 
            list of augmented bounding boxes won't contain that bounding box.
            **Suggested**: 200""")
    min_visibility = st.sidebar.slider(
        "Minimum visibility", 0.01, 1.0, min_visibility, 0.01, key='bbox_min_visibility',
        help="""**Minimum visibility** is a value between 0 and 1. If the ratio of
            the bounding box area after augmentation to the area of the 
            bounding box before augmentation becomes smaller than 
            **Minimum visibility**, the transform will drop that box. So if the 
            augmentation process cuts the most of the bounding box, that 
            box won't be present in the returned list of the augmented 
            bounding boxes. **Suggested**: 0.1""")
    # store them to use for update DB
    session_state.augmentation_config.min_area = min_area
    session_state.augmentation_config.min_visibility = min_visibility
    return min_area, min_visibility


def show_docstring(obj_with_ds):
    st.subheader("Information (Docstring) for Transformation: " +
                 obj_with_ds.__class__.__name__)
    st.text(obj_with_ds.__doc__)
