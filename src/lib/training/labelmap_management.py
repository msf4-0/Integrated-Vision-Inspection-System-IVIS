"""
Title: Labelmap Generator
Date: 8/9/2021
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
from typing import IO, List, Union
from enum import IntEnum

import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state

SRC = Path(__file__).resolve().parents[3]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

from core.utils.file_handler import get_member
# >>>> User-defined Modules >>>>
from core.utils.form_manager import remove_newline_trailing_whitespace
from core.utils.helper import split_string, get_identifier_str_IntEnum
from core.utils.log import log_error, log_info  # logger
from data_manager.database_manager import init_connection
from deployment.deployment_management import (COMPUTER_VISION_LIST, Deployment,
                                              DeploymentType)
from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2
# ************** TensorFLow ***************************
from object_detection.protos.string_int_label_map_pb2 import (
    StringIntLabelMap, StringIntLabelMapItem)
from object_detection.utils.label_map_util import (
    _validate_label_map, convert_label_map_to_categories,
    create_categories_from_labelmap)
from path_desc import chdir_root


# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration <<<<


class Framework(IntEnum):
    TensorFlow = 0
    PyTorch = 1
    Scikit_learn = 2
    Caffe = 3
    MXNet = 4
    ONNX = 5
    YOLO = 6

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return Framework[s]
        except KeyError:
            raise ValueError()


# Names to for file storing Labelmaps
LABELMAP_NAME = {
    Framework.TensorFlow: 'labelmap.pbtxt',
    Framework.PyTorch: 'labelmap.json',  # pickle,json or txt
    Framework.Scikit_learn: '',
    Framework.Caffe: '',
    Framework.MXNet: '',
    Framework.ONNX: '',
    Framework.YOLO: 'labelmap.txt',

}
FRAMEWORK = {
    "TensorFlow": Framework.TensorFlow,
    "PyTorch": Framework.PyTorch,
    "Scikit-learn": Framework.Scikit_learn,
    "Caffe": Framework.Caffe,
    "MXNet": Framework.MXNet,
    "ONNX": Framework.ONNX,
    "YOLO": Framework.YOLO
}
# <<<< Variable Declaration <<<<


def get_framework(framework: Union[str, Framework], string: bool = False) -> Union[str, Framework]:
    """Get Framework string or IntEnum constants

    Args:
        framework (Union[str, Framework]): Framework string or IntEnum constant
        string (bool, optional): True if to obtain type string, False to obtain IntEnum constant. Defaults to False.

    Returns:
        Union[str, Framework]: Converted Framework
    """

    assert isinstance(
        framework, (str, Framework)), f"framework must be String or IntEnum"

    model_type = get_identifier_str_IntEnum(
        framework, Framework, FRAMEWORK, string=string)

    return model_type


class Labels:

    def __init__(self) -> None:
        self.filename: str = None
        self.filePath: Path = None
        self.dict: List = []
        self.label_map_string:str=None

    @staticmethod
    def generate_list_of_labels(comma_separated_string: str) -> List:
        """Generate a List of labels from comma-separated string

        Args:
            comma_separated_string (str): Comma-separated string

        Returns:
            List: List of labels
        """

        labels_list = set(split_string(
            remove_newline_trailing_whitespace(str(comma_separated_string)),
            separator=','))

        return labels_list

    def generate_labelmap_string(labels_list: List[str], framework: Union[str, Framework], deployment_type: Union[str, DeploymentType]):
        """Generate String of labelmap based on Framework and Deep Learning Architectures*
        #### Currently used in Computer Vision Applications

        Args:
            labels_list (List[str]): [description]
            framework (Union[str, Framework]): [description]
            deployment_type (Union[str, DeploymentType]): [description]

        Returns:
            [type]: [description]
        """
        # Generate string of labelmap
        # strip comma separated string

        framework = get_framework(framework=framework,
                                  string=False)
        deployment_type = Deployment.get_deployment_type(deployment_type=deployment_type,
                                                         string=False)

        if framework == Framework.TensorFlow:

            if deployment_type in COMPUTER_VISION_LIST:
                labelmap_string = TensorFlow.label_map_to_text(labels_list)

                return labelmap_string

    @staticmethod
    def generate_labelmap_file(labelmap_string: str,
                               dst: Union[str, Path],
                               framework: Union[str, Framework],
                               deployment_type: Union[str, DeploymentType]
                               ) -> bool:

        framework = get_framework(framework=framework,
                                  string=False)
        deployment_type = Deployment.get_deployment_type(deployment_type=deployment_type,
                                                         string=False)
        dst = Path(dst)

        assert dst.is_dir(
        ), f"Destination directory is not found for {str(dst)} "

        if framework == Framework.TensorFlow:

            if deployment_type in COMPUTER_VISION_LIST:
                filepath = dst / 'labelmap.pbtxt'
                TensorFlow.label_map_to_pbtxt(labelmap_text=labelmap_string,
                                              filepath=filepath)

    @staticmethod
    def get_labelmap_member_from_archived(name: str, archived_filepath: Path = None, file_object: IO = None, decode: str = 'utf-8'):
        label_map_string = get_member(
            name=name, archived_filepath=archived_filepath, file_object=file_object, decode=decode)
        return label_map_string

    @staticmethod
    def generate_labelmap_dict(label_map_string: str, framework: Union[str, Framework]) -> List:
        label_map_dict = []
        framework = get_framework(framework=framework,
                                  string=False)
        if framework == Framework.TensorFlow:
            label_map_dict = TensorFlow.read_labelmap_file(
                label_map_string=label_map_string)

        return label_map_dict


class TensorFlow(object):

    @staticmethod
    def label_map_to_text(classes: List, start=1) -> str:
        # 'id' must start from 1
        msg = StringIntLabelMap()
        for id, name in enumerate(classes, start=start):
            msg.item.append(StringIntLabelMapItem(
                id=id, name=name))

        text = str(text_format.MessageToBytes(msg, as_utf8=True), 'utf-8')
        return text

    @staticmethod
    def label_map_to_pbtxt(labelmap_text: bytes, filepath: Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w", encoding="utf-8") as f:
            f.write(labelmap_text)

    @staticmethod
    def read_labelmap_file(label_map_path: Union[str, Path] = None, label_map_string: str = None, use_display_name: bool = True) -> List:

        if label_map_path:
            label_map_dict = create_categories_from_labelmap(
                label_map_path=str(label_map_path))

        elif label_map_string:
            label_map = TensorFlow.load_labelmap(
                label_map_string=label_map_string)
            max_num_classes = max(item.id for item in label_map.item)
            label_map_dict = convert_label_map_to_categories(label_map=label_map,
                                                             max_num_classes=max_num_classes,
                                                             use_display_name=use_display_name)

        else:
            label_map_dict = None

        return label_map_dict

    @staticmethod
    def load_labelmap(label_map_string: str):
        label_map = StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)
        _validate_label_map(label_map)

        return label_map
