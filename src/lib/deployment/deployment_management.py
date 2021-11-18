"""
Title: Deployment Management
Date: 28/7/2021
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
from __future__ import annotations

from collections import namedtuple
from functools import partial
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union, List, Dict, TYPE_CHECKING
from colorutils import static
import cv2
import numpy as np
from psycopg2 import sql
from PIL import Image
from time import perf_counter, sleep
from enum import IntEnum
import json
from copy import copy, deepcopy
import pandas as pd
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state
import tensorflow as tf
from yaml import full_load
from machine_learning.command_utils import export_tfod_savedmodel, run_command

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

# >>>> User-defined Modules >>>>
from path_desc import MQTT_CONFIG_PATH, TFOD_DIR, chdir_root, MEDIA_ROOT
from core.utils.log import logger
from core.utils.helper import Timer, get_identifier_str_IntEnum
from data_manager.database_manager import init_connection, db_fetchone, db_no_fetch, db_fetchall
from core.utils.form_manager import reset_page_attributes
from machine_learning.utils import classification_predict, get_test_images_labels, load_keras_model, load_labelmap, load_tfod_model, preprocess_image, segmentation_predict, tfod_detect
if TYPE_CHECKING:
    from machine_learning.trainer import Trainer
    from training.model_management import Model
from machine_learning.visuals import draw_tfod_bboxes, get_colored_mask_image

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration >>>>

# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


class DeploymentType(IntEnum):
    Image_Classification = 1
    OD = 2
    Instance = 3
    Semantic = 4

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return DeploymentType[s]
        except KeyError:
            raise ValueError()


class DeploymentPagination(IntEnum):
    Models = 0
    Deployment = 1

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return DeploymentPagination[s]
        except KeyError:
            raise ValueError()


# KIV
DEPLOYMENT_TYPE = {
    "Image Classification": DeploymentType.Image_Classification,
    "Object Detection with Bounding Boxes": DeploymentType.OD,
    "Semantic Segmentation with Polygons": DeploymentType.Instance,
    "Semantic Segmentation with Masks": DeploymentType.Semantic
}

COMPUTER_VISION_LIST = [DeploymentType.Image_Classification, DeploymentType.OD,
                        DeploymentType.Instance, DeploymentType.Semantic]
# <<<< Variable Declaration <<<<


class BaseDeployment:
    def __init__(self) -> None:
        self.id: int = None
        self.name: str = None
        self.model_selected = None


class Deployment(BaseDeployment):
    def __init__(self,
                 deployment_type: str,
                 training_path: Dict[str, Path] = None,
                 image_size: int = None,
                 metrics: List[Callable] = None,
                 metric_names: List[str] = None,
                 class_names: List[str] = None,
                 is_uploaded: bool = False,
                 category_index: Dict[int, Dict[str, Any]] = None) -> None:
        super().__init__()
        self.training_path: Dict[str, Path] = training_path
        self.metrics: List[Callable] = metrics
        self.metric_names: List[str] = metric_names
        self.class_names: List[str] = class_names
        self.class_names_arr: np.ndarray = np.array(class_names)
        self.deployment_type: str = deployment_type
        self.image_size: int = image_size

        # THIS ONLY WORKS FOR TFOD MODEL FOR NOW
        self.is_uploaded = is_uploaded
        self.category_index = category_index

        self.deployment_list: List = self.query_deployment_list()
        self.model: tf.keras.Model = None

    @classmethod
    def from_trainer(cls, trainer: Trainer):
        deployment_type = trainer.deployment_type
        training_path = trainer.training_path
        metrics = trainer.metrics
        metric_names = trainer.metric_names
        # maybe class_names can get from labelmap directly for user-uploaded models??
        class_names = trainer.class_names
        if deployment_type != 'Object Detection with Bounding Boxes':
            image_size = trainer.training_param['image_size']
        else:
            image_size = None
        return cls(deployment_type, training_path, image_size,
                   metrics, metric_names, class_names)

    @classmethod
    def from_uploaded_model(cls, model: Model, uploaded_model_dir: Path,
                            labelmap_path: Path):
        """This only works for uploaded TFOD model for now"""
        deployment_type = model.deployment_type
        training_path = {'uploaded_model_dir': uploaded_model_dir}
        category_index = load_labelmap(labelmap_path)
        class_names = [d['name'] for d in category_index.values()]
        return cls(deployment_type, training_path, category_index=category_index,
                   class_names=class_names, is_uploaded=True)

    @st.cache
    def query_deployment_list(self):
        query_deployment_list_sql = """
                                    SELECT
                                        name
                                    FROM
                                        deployment_type
                                    ORDER BY
                                        id ASC;
                                    """
        deployment_list = db_fetchall(query_deployment_list_sql, conn)
        return deployment_list if deployment_list else None

    @st.cache
    def query_model_table(self, model_table) -> NamedTuple:
        schema, table = [x for x in model_table.split('.')]
        query_model_table_SQL = sql.SQL("""SELECT
                m.id AS "ID",
                m.name AS "Name",
                f.name AS "Framework",
                m.model_path AS "Model Path"
            FROM
                {table} m
                LEFT JOIN public.framework f ON f.id = m.framework_id
                where m.deployment_id = (SELECT id from public.deployment_type where name = %s);""").format(table=sql.Identifier(schema, table))
        query_model_table_vars = [self.name]
        return_all = db_fetchall(
            query_model_table_SQL, conn, query_model_table_vars, fetch_col_name=True)
        if return_all:
            project_model_list, column_names = return_all
        else:
            project_model_list = []
            column_names = []
        return project_model_list, column_names

    @staticmethod
    @st.experimental_memo
    def get_deployment_type(deployment_type: Union[str, DeploymentType], string: bool = False):

        assert isinstance(
            deployment_type, (str, DeploymentType)), f"deployment_type must be String or IntEnum"

        deployment_type = get_identifier_str_IntEnum(
            deployment_type, DeploymentType, DEPLOYMENT_TYPE, string=string)

        return deployment_type

    def run_preparation_pipeline(self, re_export: bool = True):
        paths = self.training_path
        if self.deployment_type == 'Object Detection with Bounding Boxes':
            if self.is_uploaded:
                # this is a user-uploaded model
                # category_index has already been loaded in `self.from_uploaded_model()`
                saved_model_dir = list(
                    paths['uploaded_model_dir'].rglob("saved_model"))[0]
            else:
                # this is a project model trained in our app
                if re_export:
                    export_tfod_savedmodel(paths)
                saved_model_dir = paths['export'] / 'saved_model'
                self.category_index = load_labelmap(
                    paths['labelmap_file'])

            self.model = load_tfod_model(saved_model_dir)
        else:
            self.model = load_keras_model(
                paths['output_keras_model_file'],
                self.metrics)

        if self.deployment_type == 'Image Classification':
            *_, encoded_label_dict = get_test_images_labels(
                paths['test_set_pkl_file'],
                self.deployment_type)
            self.encoded_label_dict = encoded_label_dict

    def classification_inference_pipeline(
            self, img: np.ndarray, disable_timer: bool = True,
            **kwargs) -> Tuple[str, float]:
        with Timer("Inference time", disable_timer):
            preprocessed_img = preprocess_image(img, self.image_size)
            y_pred, y_proba = classification_predict(
                preprocessed_img,
                self.model,
                return_proba=True)
        pred_classname = self.encoded_label_dict[y_pred]
        return pred_classname, y_proba

    def tfod_inference_pipeline(
            self, img: np.ndarray, conf_threshold: float = 0.6,
            disable_timer: bool = True, draw_result: bool = True,
            **kwargs):
        with Timer("Inference time", disable_timer):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detections = tfod_detect(self.model, img,
                                     tensor_dtype=tf.uint8)
        if draw_result:
            draw_tfod_bboxes(detections, img,
                             self.category_index,
                             conf_threshold)
            return img, detections
        return detections

    def segment_inference_pipeline(
            self, img: np.ndarray, disable_timer: bool = True,
            draw_result: bool = True, class_colors: np.ndarray = None,
            ignore_background: bool = False, **kwargs):
        with Timer("Inference time", disable_timer):
            orig_H, orig_W = img.shape[:2]
            preprocessed_img = preprocess_image(img, self.image_size)
            pred_mask = segmentation_predict(
                self.model, preprocessed_img, orig_W, orig_H)
        if draw_result:
            assert class_colors is not None
            drawn_output = get_colored_mask_image(
                img, pred_mask, class_colors,
                ignore_background=ignore_background)
            return drawn_output, pred_mask
        return pred_mask

    def get_inference_pipeline(self, **kwargs) -> Callable:
        assert 'img' not in kwargs, "Image should only be passed in during inference time"
        if self.deployment_type == 'Image Classification':
            return partial(self.classification_inference_pipeline, **kwargs)
        elif self.deployment_type == 'Object Detection with Bounding Boxes':
            return partial(self.tfod_inference_pipeline, **kwargs)
        elif self.deployment_type == 'Semantic Segmentation with Polygons':
            return partial(self.classification_inference_pipeline, **kwargs)

    def get_detection_results(self, detections: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        for class_idx, prob in zip(detections['detection_classes'],
                                   detections['detection_scores']):
            detection = {'name': self.category_index[class_idx]['name'],
                         'prob': prob}
            results.append(detection)
        return results

    def get_segmentation_results(self, prediction_mask: np.ndarray) -> Dict[str, str]:
        class_names = self.class_names_arr[np.unique(prediction_mask)]
        results = {'Detected classes': class_names}
        return results

    @staticmethod
    def reset_deployment_page():
        """Method to reset all widgets and attributes in the Deployment Pages when changing pages
        """

        # probably should not reset "client" to let it stay connected?
        project_attributes = ["deployment_pagination", "training", "trainer",
                              "training_paths", "publishing", "camera"]

        reset_page_attributes(project_attributes)


def main():
    print("Hi")


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
