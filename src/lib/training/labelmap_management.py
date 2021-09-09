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
from typing import List, Union

import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state


SRC = Path(__file__).resolve().parents[3]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

# >>>> User-defined Modules >>>>
from core.utils.form_manager import remove_newline_trailing_whitespace
from core.utils.helper import split_string
from core.utils.log import log_error, log_info  # logger
from data_manager.database_manager import init_connection
from deployment.deployment_management import COMPUTER_VISION_LIST, Deployment, DeploymentType
from path_desc import chdir_root

from training.model_management import Framework, Model

# ************** TensorFLow ***************************
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from google.protobuf import text_format
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration <<<<

# <<<< Variable Declaration <<<<


class Labels:

    def __init__(self) -> None:
        pass

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

        framework = Model.get_framework(framework=framework,
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

        framework = Model.get_framework(framework=framework,
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
