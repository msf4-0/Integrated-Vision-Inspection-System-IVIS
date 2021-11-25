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
import shutil
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import partial
import sys
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union, List, Dict, TYPE_CHECKING
from enum import IntEnum
import json

from psycopg2 import sql
import cv2
import numpy as np
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state
import tensorflow as tf

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

# >>>> User-defined Modules >>>>
from core.utils.log import logger
from core.utils.helper import Timer, get_identifier_str_IntEnum, get_now_string
from data_manager.database_manager import init_connection, db_fetchone, db_no_fetch, db_fetchall
from core.utils.form_manager import reset_page_attributes
from machine_learning.utils import classification_predict, get_test_images_labels, load_keras_model, load_labelmap, load_tfod_model, preprocess_image, segmentation_predict, tfod_detect
if TYPE_CHECKING:
    from machine_learning.trainer import Trainer
    from training.model_management import Model
from machine_learning.visuals import draw_tfod_bboxes, get_colored_mask_image
from machine_learning.command_utils import export_tfod_savedmodel
from deployment.utils import reset_camera, reset_camera_ports, reset_client, reset_csv_file_and_writer, reset_record_and_vid_writer
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
    UploadModel = 1
    Deployment = 2
    SwitchUser = 3

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return DeploymentPagination[s]
        except KeyError:
            raise ValueError()


@dataclass(eq=False)
class DeploymentConfig:
    input_type: str = 'Image'
    video_width: int = 500
    use_camera: bool = False
    camera_type: str = 'USB Camera'
    camera_port: int = 0
    retention_period: int = 7
    mqtt_qos: int = 1
    publishing: bool = True

    # not always used
    ip_cam_address: str = ''
    confidence_threshold: float = 0.4
    ignore_background: bool = False

    def asdict(self):
        return deepcopy(self.__dict__)


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
                 project_path: Path,
                 deployment_type: str,
                 training_path: Dict[str, Path] = None,
                 image_size: int = None,
                 metrics: List[Callable] = None,
                 metric_names: List[str] = None,
                 class_names: List[str] = None,
                 is_uploaded: bool = False,
                 category_index: Dict[int, Dict[str, Any]] = None) -> None:
        super().__init__()
        self.project_path = project_path
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

        # not needed for now
        # self.deployment_list: List = self.query_deployment_list()
        self.model: tf.keras.Model = None

    @classmethod
    def from_trainer(cls, trainer: Trainer):
        project_path = trainer.project_path
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
        return cls(project_path, deployment_type, training_path, image_size,
                   metrics, metric_names, class_names)

    @classmethod
    def from_uploaded_model(cls, model: Model, uploaded_model_dir: Path,
                            category_index: Dict[int, Dict[str, Any]]):
        """This only works for uploaded TFOD model for now"""
        project_path = session_state.project.get_project_path(
            session_state.project.name)
        deployment_type = model.deployment_type
        training_path = {'uploaded_model_dir': uploaded_model_dir}
        class_names = [d['name'] for d in category_index.values()]
        return cls(project_path, deployment_type, training_path,
                   category_index=category_index, class_names=class_names,
                   is_uploaded=True)

    @staticmethod
    @st.experimental_memo
    def query_deployment_list():
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
                saved_model_dir = next(
                    paths['uploaded_model_dir'].rglob("saved_model"))
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
            self.encoded_label_dict: Dict[int, str] = encoded_label_dict

    def classification_inference_pipeline(
            self, img: np.ndarray, **kwargs) -> Tuple[str, float]:
        preprocessed_img = preprocess_image(img, self.image_size)
        y_pred, y_proba = classification_predict(
            preprocessed_img,
            self.model,
            return_proba=True)
        pred_classname = self.encoded_label_dict[y_pred]
        return pred_classname, y_proba

    def tfod_inference_pipeline(
            self, img: np.ndarray, conf_threshold: float = 0.6,
            draw_result: bool = True, **kwargs):
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
            self, img: np.ndarray,
            draw_result: bool = True, class_colors: np.ndarray = None,
            ignore_background: bool = False, **kwargs):
        orig_H, orig_W = img.shape[:2]
        preprocessed_img = preprocess_image(img, self.image_size)
        pred_mask = segmentation_predict(
            self.model, preprocessed_img, orig_W, orig_H)
        if draw_result:
            # must provide class_colors
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

    def get_detection_results(self, detections: Dict[str, Any],
                              conf_threshold: float = None) -> List[Dict[str, Any]]:
        results = []
        now = get_now_string()
        for class_id, prob, box in zip(detections['detection_classes'],
                                       detections['detection_scores'],
                                       detections['detection_boxes']):
            if prob < conf_threshold:
                continue
            ymin, xmin, ymax, xmax = np.around(box, 4).tolist()
            tl = [xmin, ymin]
            br = [xmax, ymax]
            detection = {'name': self.category_index[class_id]['name'],
                         # need to change to string to be serialized with json.dumps()
                         'probability': f"{prob * 100:.2f}%",
                         'top_left': tl,
                         'bottom_right': br,
                         'time': now}
            results.append(detection)
        return results

    def get_segmentation_results(self, prediction_mask: np.ndarray) -> List[Dict[str, Any]]:
        class_names = self.class_names_arr[np.unique(prediction_mask)]
        results = [{'classes_found': class_names,
                   'time': get_now_string()}]
        return results

    def get_csv_path(self, now: datetime) -> Path:
        full_date = now.strftime("%d-%b-%Y")
        csv_path = self.project_path / 'deployment_results' \
            / full_date / f"{full_date}.csv"
        return csv_path

    @staticmethod
    def get_datetime_from_csv_path(csv_path: Path) -> datetime:
        # year_and_month = csv_path.parent.name
        # date_today = csv_path.stem
        # file_date = f"{year_and_month}_{date_today}"
        full_date = csv_path.stem
        dt_format = "%d-%b-%Y"  # based on get_csv_path()
        dt = datetime.strptime(full_date, dt_format)
        return dt

    def delete_old_csv_files(self, retention_period: int):
        """Delete CSV files older than `retention_period` (in `days`)."""
        # directory based on get_csv_path()
        csv_dir = self.project_path / 'deployment_results'
        csv_paths = csv_dir.iterdir()
        sorted_csv_paths = sorted(
            csv_paths, key=self.get_datetime_from_csv_path, reverse=True)
        now = datetime.now()
        for p in sorted_csv_paths:
            csv_date = self.get_datetime_from_csv_path(p)
            days_from_created = (now - csv_date).days
            if days_from_created > retention_period:
                logger.info(f"Removing old CSV file older than {retention_period} "
                            f"days at {p}")
                shutil.rmtree(p)
            else:
                # stop because no more older files
                break

    @staticmethod
    def reset_deployment_page():
        """Method to reset all widgets and attributes in the Deployment Pages when changing pages
        """
        tf.keras.backend.clear_session()
        if 'csv_file' in session_state and not session_state.csv_file.closed:
            session_state.csv_file.close()

        reset_camera()
        reset_camera_ports()
        reset_record_and_vid_writer()
        reset_csv_file_and_writer()
        reset_client()

        project_attributes = ["deployment_pagination", "trainer", "publishing",
                              "refresh", "deployment_conf", "today"]

        reset_page_attributes(project_attributes)


def main():
    print("Hi")


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
