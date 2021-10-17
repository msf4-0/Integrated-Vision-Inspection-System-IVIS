import copy
import os
import re
import subprocess
import sys
import pprint
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from imutils.paths import list_images
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import glob
import pandas as pd
import albumentations as A

import streamlit as st
from streamlit import session_state
from streamlit_tensorboard import st_tensorboard
from stqdm import stqdm

from object_detection.utils import label_map_util
from .visuals import pretty_st_metric, PrettyMetricPrinter

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

# >>>> User-defined Modules >>>>
from core.utils.log import logger


def run_tensorboard(logdir: Path, port: int = 6007, width: int = 1080):
    # TODO: test whether this TensorBoard works after deployed the app
    logger.info(f"Running TensorBoard on {logdir}")
    # NOTE: this st_tensorboard does not work if the path passed in
    #  is NOT in POSIX format, thus the `as_posix()` method to convert
    #  from WindowsPath to POSIX format to work in Windows
    st_tensorboard(logdir=logdir.as_posix(),
                   port=port, width=width, scrolling=True)


def find_tfod_metric(name: str, cmd_output: str) -> Tuple[str, Union[int, float]]:
    """
    Find the specific metric name in the command output (`cmd_output`)
    and returns only the digits (i.e. values) of the metric.

    Basically search using regex groups, then take the last match,
    then take the first group for the metric name, and the second group for the digits.
    """
    assert name in ('Step', 'Loss')
    try:
        if name == 'Step':
            value = re.findall(f'({name})\s+(\d+)', cmd_output)[-1][1]
            return name, int(value)
        else:
            loss_name, value = re.findall(
                f'{name}/(\w+).+(\d+\.\d+)', cmd_output)[-1]
            return loss_name, float(value)
    except IndexError:
        logger.error(f"Value for '{name}' not found from {cmd_output}")


def run_command(command_line_args: str, st_output: bool = False,
                stdout_output: bool = True,
                filter_by: Optional[List[str]] = None,
                pretty_print: Optional[bool] = False,
                ) -> Union[str, None]:
    """
    Running commands or scripts, while getting live stdout

    Args:
        command_line_args (str): Command line arguments to run.
        st_output (bool, optional): Set `st_output` to True to
            show the console outputs LIVE on Streamlit. Defaults to False.
        stdout_output (bool, optional): Set `stdout_output` to True to
            show the console outputs LIVE on terminal. Defaults to True.
        filter_by (Optional[List[str]], optional): Provide `filter_by` to
            filter out other strings and show the `filter_by` strings on Streamlit app. Defaults to None.
    Returns:
        str: The entire console output from the command after finish running.
    """
    if pretty_print:
        command_line_args = pprint.pformat(command_line_args)
    logger.info(f"Running command: '{command_line_args}'")
    # shell=True to work on String instead of list
    process = subprocess.Popen(command_line_args, shell=True,
                               # stdout to capture all output
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               # text to directly decode the output
                               text=True)
    if filter_by:
        output_str_list = []
    for line in process.stdout:
        # remove empty trailing spaces, and also string with only spaces
        line = line.strip()
        if line and st_output:
            if filter_by:
                for filter_str in filter_by:
                    if filter_str in line:
                        st.markdown(line)
                        output_str_list.append(line)
            else:
                st.markdown(line)
        if line and stdout_output:
            print(line)
    process.wait()
    if filter_by and not output_str_list:
        # there is nothing obtained from the filtered stdout
        logger.error("Error with training!")
        st.error("Some error has occurred during training ..."
                 " Please try again")
        time.sleep(3)
        st.experimental_rerun()
    elif filter_by:
        return '\n'.join(output_str_list)
    return process.stdout


def run_command_update_metrics(
    command_line_args: str,
    stdout_output: bool = True,
    step_name: str = 'Step',
    metric_names: Tuple[str] = None,
) -> str:
    # ! DEPRECATED, Using run_command_update_metrics_2 function
    """
    Running commands or scripts, while getting live stdout

    Args:
        command_line_args (str): Command line arguments to run.
        st_output (bool, optional): Set `st_output` to True to
            show the console outputs LIVE on Streamlit. Defaults to False.
        filter_by (Optional[List[str]], optional): Provide `filter_by` to
            filter out other strings and show the `filter_by` strings on Streamlit app. Defaults to None.
    Returns:
        str: The entire console output from the command after finish running.
    """
    assert metric_names is not None, "Please pass in metric_names to use for search and updates"

    logger.info(f"Running command: '{command_line_args}'")
    process = subprocess.Popen(command_line_args, shell=True,
                               # stdout to capture all output
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               # text to directly decode the output
                               text=True)

    prev_progress = session_state.new_training.progress
    curr_metrics = session_state.new_training.training_model.metrics
    curr_metrics = copy.copy(curr_metrics)
    prev_metrics = copy.copy(curr_metrics)
    for line in process.stdout:
        # remove empty trailing spaces, and also string with only spaces
        line = line.strip()
        if line:
            if step_name in line:
                step_val = find_tfod_metric(step_name, line)
                # only update the `Step` progress if it's different
                if step_val and step_val != prev_progress[step_name]:
                    # update previous step value to be the current
                    prev_progress[step_name] = step_val
                    session_state.new_training.update_progress(
                        prev_progress, verbose=True)
                    st.markdown(f"**{step_name}**: {step_val}")
            for name in metric_names:
                if name in line:
                    metric_val = find_tfod_metric(name, line)
                    if metric_val:
                        curr_metrics[name] = metric_val
                        num_same_values = np.sum(np.isclose(
                            list(curr_metrics.values()),
                            list(prev_metrics.values())))
                        print(f"{num_same_values}")
                        # only update and show the metrics when all values
                        #  are already updated with the latest different metrics
                        if num_same_values == 0:
                            session_state.new_training.update_metrics(
                                curr_metrics, verbose=True)
                            # show the nicely formatted metrics on Streamlit
                            pretty_st_metric(
                                curr_metrics,
                                prev_metrics,
                                float_format='.3g')
                            # update previous metrics to be the current
                            prev_metrics = copy.copy(curr_metrics)
        if line and stdout_output:
            print(line)
    process.wait()
    return process.stdout


def run_command_update_metrics_2(
    command_line_args: str,
    stdout_output: bool = True,
    step_name: str = 'Step',
    pretty_print: Optional[bool] = False,
    # metric_names: Tuple[str] = None,
) -> str:
    """[summary]

    Args:
        command_line_args (str): Command line arguments to run.
        stdout_output (bool, optional): Set `stdout_output` to True to
            show the console outputs LIVE on terminal. Defaults to True.
        step_name (str, optional): The key name used to store our training step progress.
            Should be 'Step' for now. Defaults to 'Step'.
        # ! Deprecated
        metric_names (Tuple[str], optional): The metric names to search for and update our
            Model instance and database. Must be passed in. Defaults to None.

    Returns:
        str: the entire console output generated from the TFOD training script
    """
    # assert metric_names is not None, "Please pass in metric_names to use for search and updates"
    if pretty_print:
        command_line_args = pprint.pformat(command_line_args)
    logger.info(f"Running command: '{command_line_args}'")
    process = subprocess.Popen(command_line_args, shell=True,
                               # stdout to capture all output
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               # text to directly decode the output
                               text=True)

    pretty_metric_printer = PrettyMetricPrinter()
    # list of stored stdout lines
    current_lines = []
    # flag to track the 'Step' text
    step_name_found = False
    for line in process.stdout:
        # remove empty trailing spaces, and also string with only spaces
        line = line.strip()
        if stdout_output and line:
            # print to console
            print(line)
        if line:
            # keep storing when stored stdout lines are less than 12,
            #  12 is the exact number of lines output by the script
            #  that will contain all the metrics after the line
            #  with text 'Step', e.g.:
            """
            INFO:tensorflow:Step 900 per-step time 0.561s
            I0813 13:07:28.326545 139687560058752 model_lib_v2.py:700] Step 900 per-step time 0.561s
            INFO:tensorflow:{'Loss/classification_loss': 0.34123567,
            'Loss/localization_loss': 0.0464548,
            'Loss/regularization_loss': 0.24863605,
            'Loss/total_loss': 0.6363265,
            'learning_rate': 0.025333151}
            I0813 13:07:28.326872 139687560058752 model_lib_v2.py:701] {'Loss/classification_loss': 0.34123567,
            'Loss/localization_loss': 0.0464548,
            'Loss/regularization_loss': 0.24863605,
            'Loss/total_loss': 0.6363265,
            'learning_rate': 0.025333151}
            """
            if step_name in line:
                step_name_found = True
            if step_name_found:
                current_lines.append(line)
            if len(current_lines) == 12:
                # take the first line because that's where `step_name` was found and appended first
                _, step_val = find_tfod_metric(step_name, current_lines[0])
                session_state.new_training.progress[step_name] = step_val
                session_state.new_training.update_progress(
                    session_state.new_training.progress, verbose=True)
                st.markdown(f"**{step_name}**: {step_val}")

                metrics = {}
                # for name in metric_names:
                for line in current_lines[2:]:
                    metric = find_tfod_metric('Loss', line)
                    if metric:
                        metric_name, metric_val = metric
                        metrics[metric_name] = metric_val
                session_state.new_training.update_metrics(
                    metrics, verbose=True)

                # show the nicely formatted metrics on Streamlit
                pretty_metric_printer.write(metrics)
                st.markdown("___")

                # reset the list of outputs to start storing again
                current_lines = []
                step_name_found = False
    process.wait()
    return process.stdout


def load_image_into_numpy_array(path: str):
    """Load an image from file into a numpy array.
    Puts image into numpy array of shape (height, width, channels), where channels=3 for RGB to feed into tensorflow graph.
    Args:
    path: the file path to the image
    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    img = cv2.imread(path)
    # convert from OpenCV's BGR to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.asarray(img)


def load_labelmap(labelmap_path):
    """
    Returns:
    category_index =
    {
        1: {'id': 1, 'name': 'category_1'},
        2: {'id': 2, 'name': 'category_2'},
        3: {'id': 3, 'name': 'category_3'},
        ...
    }
    """
    category_index = label_map_util.create_category_index_from_labelmap(
        labelmap_path,
        use_display_name=True)
    return category_index


@st.experimental_memo
def xml_to_df(path: str) -> pd.DataFrame:
    """
    If a path to XML file is passed in, parse it directly.
    If directory is passed in, iterates through all .xml files (generated by labelImg) 
    in a given directory and combines
    them in a single Pandas dataframe.

    Parameters:
    ----------
    path : str
        The path containing the .xml files
    Returns
    -------
    Pandas DataFrame
        The produced dataframe
    """
    if isinstance(path, Path):
        path = str(path)

    xml_list = []

    if os.path.isfile(path):
        xml_files = [path]
    else:
        xml_files = glob.glob(path + "/*.xml")

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find("filename").text
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)
        for member in root.findall("object"):
            bndbox = member.find("bndbox")
            value = (
                filename,
                width,
                height,
                member.find("name").text,
                int(bndbox.find("xmin").text),
                int(bndbox.find("ymin").text),
                int(bndbox.find("xmax").text),
                int(bndbox.find("ymax").text),
            )
            xml_list.append(value)
    column_name = [
        "filename",
        "width",
        "height",
        "classname",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def get_bbox_label_info(xml_df: pd.DataFrame,
                        image_name: str) -> Tuple[List[str], Tuple[int, int, int, int]]:
    """Get the class name and bounding box coordinates associated with the image."""
    annot_df = xml_df.loc[xml_df['filename'] == image_name]
    class_names = annot_df['classname'].values
    bboxes = annot_df.loc[:, 'xmin': 'ymax'].values
    return class_names, bboxes


def get_transform():
    """Get the Albumentations' transform using the existing augmentation config stored in DB."""
    existing_aug = session_state.new_training.augmentation_config.augmentations

    transform_list = []
    for transform_name, param_values in existing_aug.items():
        transform_list.append(getattr(A, transform_name)(**param_values))

    if session_state.project.deployment_type == 'Object Detection with Bounding Boxes':
        min_area = session_state.new_training.augmentation_config.min_area
        min_visibility = session_state.new_training.augmentation_config.min_visibility
        transform = A.Compose(
            transform_list,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=min_area,
                min_visibility=min_visibility,
                label_fields=['class_names']
            ))
    else:
        transform = A.Compose(transform_list)
    return transform


def generate_tfod_xml_csv(image_paths: List[str],
                          xml_dir: Path,
                          output_img_dir: Path,
                          csv_path: Path,
                          train_size: int):
    """Generate TFOD's CSV file for augmented images and bounding boxes used for generating TF Records.
    Also save the transformed images to the `output_img_dir` at the same time."""

    output_img_dir.mkdir(parents=True, exist_ok=True)

    transform = get_transform()
    xml_df = xml_to_df(str(xml_dir))

    if train_size > len(image_paths):
        # randomly select the remaining paths and extend them to the original List
        # to make sure to go through the entire dataset for at least once
        n_remaining = train_size - len(image_paths)
        image_paths.extend(np.random.choice(
            image_paths, size=n_remaining, replace=True))

    logger.info('Generating CSV file for augmented bounding boxes ...')
    start = time.perf_counter()
    xml_list = []
    for image_path in stqdm(image_paths):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        filename = os.path.basename(image_path)
        class_names, bboxes = get_bbox_label_info(xml_df, filename)
        width, height = xml_df.loc[xml_df['filename'] == filename,
                                   'width': 'height'].values[0]

        transformed = transform(image=image, bboxes=bboxes,
                                class_names=class_names)
        transformed_image = transformed['image']
        # also save the transformed image at the same time to avoid doing it again later
        cv2.imwrite(str(output_img_dir / filename), transformed_image)

        transformed_bboxes = np.array(transformed['bboxes'], dtype=np.int32)
        transformed_class_names = transformed['class_names']

        for bbox, class_name in zip(transformed_bboxes, transformed_class_names):
            value = (
                filename,
                width,
                height,
                class_name,
                *bbox
            )
            xml_list.append(value)

        col_names = [
            "filename",
            "width",
            "height",
            "classname",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
        ]

    xml_df = pd.DataFrame(xml_list, columns=col_names)
    xml_df.to_csv(csv_path)
    time_elapsed = time.perf_counter() - start
    logger.info(f"Done. {time_elapsed = :.4f} seconds")


def get_augmented_data(image_paths: Path, mask_paths: Path = None):
    transform = get_transform()
