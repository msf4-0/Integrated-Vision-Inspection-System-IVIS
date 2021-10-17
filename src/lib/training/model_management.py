"""
Title: Model Management
Date: 20/7/2021
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

import json
import sys
from collections import namedtuple
from datetime import datetime
from enum import IntEnum
from logging import error
from os import name
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import pandas as pd
import psycopg2
import streamlit as st
from PIL import Image
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state
from streamlit.uploaded_file_manager import UploadedFile

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

from core.utils.code_generator import get_random_string
from core.utils.file_handler import (create_folder_if_not_exist, list_files_in_archived,
                                     save_uploaded_extract_files)
from core.utils.form_manager import (check_if_exists, check_if_field_empty,
                                     reset_page_attributes)
from core.utils.helper import (create_dataframe, dataframe2dict,
                               datetime_formatter, get_dataframe_row,
                               get_directory_name, get_identifier_str_IntEnum)
from core.utils.log import logger  # logger
from data_manager.database_manager import (db_fetchall, db_fetchone,
                                           db_no_fetch, init_connection)
from deployment.deployment_management import (COMPUTER_VISION_LIST, DEPLOYMENT_TYPE, Deployment,
                                              DeploymentType)
# >>>> User-defined Modules >>>>
from path_desc import (PRE_TRAINED_MODEL_DIR, PROJECT_DIR,
                       USER_DEEP_LEARNING_MODEL_UPLOAD_DIR, chdir_root,
                       TFOD_MODELS_TABLE_PATH, CLASSIF_MODELS_NAME_PATH,
                       SEGMENT_MODELS_TABLE_PATH)

from training.labelmap_management import Labels

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])

# >>>> Variable Declaration >>>>


class ModelType(IntEnum):
    PreTrained = 0
    ProjectTrained = 1
    UserUpload = 2

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return ModelType[s]
        except KeyError:
            raise ValueError()


MODEL_TYPE = {
    "Pre-trained Models": ModelType.PreTrained,
    "Project Models": ModelType.ProjectTrained,
    "User Deep Learning Model Upload": ModelType.UserUpload
}


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


FRAMEWORK = {
    "TensorFlow": Framework.TensorFlow,
    "PyTorch": Framework.PyTorch,
    "Scikit-learn": Framework.Scikit_learn,
    "Caffe": Framework.Caffe,
    "MXNet": Framework.MXNet,
    "ONNX": Framework.ONNX,
    "YOLO": Framework.YOLO
}
# **************************************************************************************
# `.pkl` => Pickle format to serialise weights and biases of the model graph
# `.pt` and `.pth` are also serialised model graph by PyTorch
# `.pkl` are compatible with PyTorch but .pt and .pth recommended for PyTorch
# NOTE ONLY model_extensions are COMPULSORY check !!! [~~Line 198 of user_model_upload.py]
# Others can be updated otherwise
# **************************************************************************************

MODEL_FILES = {
    Framework.TensorFlow: {
        'model_extension': ('.pb', 'h5'),
        'checkpoint': 'checkpoint',
        'config': '.config',
        'labelmap': ('labelmap.pbtxt')
    },
    Framework.PyTorch: {
        'model_extension': ('.pt', '.pth', '.pkl'),
        'checkpoint': (),
        'config': (),
        'labelmap': ('.json')  # NOTE KIV index_to_name.json
    },
    Framework.Scikit_learn: {
        'model_extension': ('.pkl', 'json'),
        'checkpoint': (),
        'config': (),
        'labelmap': ()
    },
    Framework.Caffe: {
        'model_extension': ('.caffemodel', '.pb', '.pbtxt'),
        'checkpoint': (),
        'config': (),
        'labelmap': ()
    },
    Framework.MXNet: {
        # `json` for model graph, `params` for weights and biases
        'model_extension': ('.onnx', '.json', '.params'),
        'checkpoint': (),
        'config': (),
        'labelmap': ()
    },
    Framework.ONNX: {
        'model_extension': ('.onnx'),
        'checkpoint': (),
        'config': (),
        'labelmap': ()
    }, Framework.YOLO: {
        # Varies, not a framework but architecture. Possible to be trained using other frameworks stated above
        'model_extension': (),
        'checkpoint': (),
        'config': (),
        'labelmap': ('.txt')
    }

}


class ModelsPagination(IntEnum):
    Dashboard = 0
    ExistingModels = 1
    ModelUpload = 2
    TrainingConfig = 3
    AugmentationConfig = 4
    # TrainingInfoDataset = 5  # this is an old page before models_page
    # Training = 6  # time to run training and show training progress

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return ModelsPagination[s]
        except KeyError:
            raise ValueError()


# ********************************************************************
# ****************** TODO: TO be UPDATED ****************************
# ********************************************************************
EVALUATION_TAGS = {
    DeploymentType.Image_Classification: ['Confusion Matrix', 'Accuracy', 'Precision', 'Recall', 'FLOPS'],
    DeploymentType.OD: ['COCO', 'Pascal VOC', 'Accuracy', 'Precision', 'Recall', 'FLOPS'],
    DeploymentType.Instance: ['COCO', 'Pascal VOC', 'Accuracy', 'Precision', 'Recall', 'FLOPS'],
    DeploymentType.Semantic: ['COCO', 'Pascal VOC',
                              'Accuracy', 'Precision', 'Recall', 'FLOPS']
}

# TODO #17 Work on converter


class ModelCompatibility(IntEnum):
    Compatible = 0
    MissingExtraFiles_ModelExists = 1
    MissingModel = 2
    MissingExtraFiles_MissingModel = 3

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return ModelCompatibility[s]
        except KeyError:
            raise ValueError()

# <<<< Variable Declaration <<<<


class BaseModel:
    def __init__(self, model_id: Union[int, str]) -> None:
        self.id: Union[str, int] = model_id
        self.name: str = ''
        self.desc: str = ''
        self.deployment_type: str = ''
        self.metrics: Dict = {}
        self.model_input_size: Dict = {}
        self.perf_metrics: List = []
        self.model_type: str = ''
        self.framework: str = ''
        self.training_id: int = None
        self.model_path: Path = None
        self.model_path_relative: str = ''
        self.labelmap_path: Path = None
        self.saved_model_dir: Path = None
        self.has_submitted: bool = False
        self.updated_at: datetime = None
        self.file_upload: UploadedFile = None
        self.compatibility_flag = ModelCompatibility.MissingModel
        self.labelmap = None

    @st.cache
    def get_model_path(self):
        query_model_project_training_SQL = """
                SELECT
                    p.project_path,
                    t.name
                FROM
                    public.models m
                    INNER JOIN public.training t ON m.training_id = t.id
                    INNER JOIN public.project p ON t.project_id = p.id
                WHERE
                    m.id = %s;
                        """
        query_model_project_training_vars = [self.id]
        query = db_fetchone(query_model_project_training_SQL,
                            conn, query_model_project_training_vars)

        return query

    @staticmethod
    # Wrapper for check_if_exists function from form_manager.py
    def check_if_exists(context: Dict[str, Any], conn) -> bool:
        table = 'public.models'
        exists_flag = check_if_exists(
            table, context['column_name'], context['value'], conn)

        return exists_flag

    # Wrapper for check_if_exists function from form_manager.py
    def check_if_field_empty(self, context: Dict,
                             field_placeholder: Dict,
                             name_key: str,
                             deployment_type_constant: DeploymentType = None,
                             input_size_context: Dict = {}) -> bool:
        """Check if Compulsory fields are filled and Unique information not 
        duplicated in the database

        Args:
            context (Dict): Dictionary with widget name as key and widget value as value**
            field_placeholder (Dict): Dictionary with st.empty() key as key and st.empty() object as value. 
            *Key has same name as its respective widget

            name_key (str): Key of Database row name. Used to obtain value from 'context' Dictionary.
            *Pass 'None' is not required to check row exists

            deployment_type_constant (DeploymentType, optional): DeploymentType IntEnum class constant. Defaults to None.
            input_size_context (Dict, optional): Context to check Model Input Size depending on Deployment Type (refer to `context` args ** above). Defaults to {}.

        Returns:
            bool: True if NOT EMPTY + NOT EXISTS, False otherwise.
        """

        empty_fields = []
        check_if_exists = self.check_if_exists
        empty_fields.append(check_if_field_empty(
            context, field_placeholder, name_key, check_if_exists))

        if input_size_context:
            if deployment_type_constant in COMPUTER_VISION_LIST:
                empty_fields.append(check_if_field_empty(
                    input_size_context, field_placeholder))

        sleep(0.5)

        return sorted(empty_fields)[0]

    def check_if_required_files_exist(self, uploaded_file: UploadedFile) -> bool:
        # check if necessary files required included in the package
        # Load list of files
        # Check Models
        # Check checkpoint
        # Check pipeline
        # Check labelmap
        # CHECK at submission

        framework_const = Model.get_framework(self.framework, string=False)
        deployment_type_const = Deployment.get_deployment_type(
            self.deployment_type)

        with st.spinner("Checking compatible files in uploaded model"):
            file_list = list_files_in_archived(archived_filepath=uploaded_file.name,
                                               file_object=uploaded_file)
            try:
                # Currently supports TensorFlow for Object Detection
                if framework_const == Framework.TensorFlow:
                    framework_check_list = MODEL_FILES[Framework.TensorFlow]
                    model_files = []  # use to temporariy store detected files, raise Error if length>1!!!
                    checkpoint_files = []
                    config_files = []
                    labelmap_files = []

                    for file in file_list:
                        if file.endswith(framework_check_list['model_extension']):
                            model_files.append(file)
                            logger.info(model_files)
                        assert len(
                            model_files) <= 1, "There should only be one model file"

                        if Path(file).name == framework_check_list['checkpoint']:
                            checkpoint_files.append(file)
                            logger.info(checkpoint_files)

                        if deployment_type_const in [DeploymentType.Image_Classification, DeploymentType.OD,
                                                     DeploymentType.Instance, DeploymentType.Semantic]:

                            # OPTIONAL
                            if file.endswith(framework_check_list['config']):
                                config_files.append(file)
                                logger.info(config_files)

                            if Path(file).name == framework_check_list['labelmap']:
                                labelmap_files.append(file)
                                logger.info(labelmap_files)

                    assert 0 < len(model_files) <= 1, "Model file missing"
                    assert len(
                        checkpoint_files) >= 1, "Checkpoint files missing"
                    self.compatibility_flag = ModelCompatibility.Compatible  # Set flag as Compatible

                    if deployment_type_const in COMPUTER_VISION_LIST:
                        if len(config_files) == 0:
                            st.warning(
                                f"**pipeline.config** file is missing, please include inside the archived folder as required by TensorFlow Object Detection API.")
                            logger.warning(
                                f"**pipeline.config** file is missing, please include inside the archived folder as required by TensorFlow Object Detection API.")
                            self.compatibility_flag = ModelCompatibility.MissingExtraFiles_ModelExists
                        if len(labelmap_files) == 0:
                            st.warning(
                                f"**labelmap.pbtxt** files not included in the uploaded folder. Please include for instant deployment. It is not required for new training")
                            logger.warning(
                                f"**labelmap.pbtxt** files not included in the uploaded folder. Please include for instant deployment. It is not required for new training")
                            self.compatibility_flag = ModelCompatibility.MissingExtraFiles_ModelExists

                    st.success(
                        f"**{uploaded_file.name}** contains the required files for Training")

                    return True, labelmap_files

                elif framework_const == Framework.PyTorch:
                    pass
                elif framework_const == Framework.Scikit_learn:
                    pass
                elif framework_const == Framework.Caffe:
                    pass
                elif framework_const == Framework.MXNet:
                    pass
                elif framework_const == Framework.ONNX:
                    pass

            except Exception as e:
                error_msg = f"{e}"
                logger.error(error_msg)
                st.error(error_msg)
                self.compatibility_flag = ModelCompatibility.MissingModel
                return False

    @staticmethod
    def get_model_type(model_type: Union[str, ModelType], string: bool = False) -> Union[str, ModelType]:
        """Get Model Type string or IntEnum constants

        Args:
            model_type (Union[str, ModelType]): Model Type string or IntEnum constant
            string (bool, optional): True if to obtain type string, False to obtain IntEnum constant. Defaults to False.

        Returns:
            Union[str, ModelType]: Converted Model Type
        """

        assert isinstance(
            model_type, (str, ModelType)), f"model_type must be String or IntEnum"

        model_type = get_identifier_str_IntEnum(
            model_type, ModelType, MODEL_TYPE, string=string)

        return model_type

    @staticmethod
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

    @staticmethod
    @st.cache
    def get_framework_list() -> List[NamedTuple]:
        """Get a list of framework

        Returns:
            List[NamedTuple]: List of Framework
        """
        get_framework_list_SQL = """
            SELECT
             
                name as "Name"
            FROM
                public.framework;
                    """
        framework_list = db_fetchall(get_framework_list_SQL, conn)
        return framework_list

    @staticmethod
    def get_pt_user_project_model_path(model_path: Union[str, Path] = None,
                                       framework: str = None,
                                       project_name: str = None,
                                       training_name: str = None,
                                       model_type: Union[str,
                                                         ModelType] = None,
                                       new_model_flag: bool = False,
                                       **model_row) -> Path:
        """Get directory path for Pre-trained models and User Upload Deep Learning Models

        Args:
            model_path (Union[str, Path]): Relative path to model (queried from DB) / Model Name for New model creation
            framework (str): Framework of model
            project_name (str): Project Name
            training_name (str): Training Name
            model_type (Union[str, ModelType]): Type of model
            new_model_flag (bool, optional): True is new model to be created, otherwise False. Defaults to False.

        Returns:
            Path: Path-like object of model path
        """

        if model_row:
            model_path = model_row['Model Path']
            framework = model_row['Framework']
            model_type = model_row['Model Type']
            training_name = model_row['Training Name']

        # Get IntEnum constant
        model_type = BaseModel.get_model_type(
            model_type=model_type, string=False)

        framework = get_directory_name(framework)

        model_path = get_directory_name(
            model_path) if new_model_flag else model_path  # format model_path if new model creation

        if model_type == ModelType.PreTrained:
            model_path = PRE_TRAINED_MODEL_DIR / framework / str(model_path)

        elif model_type == ModelType.UserUpload:
            model_path = USER_DEEP_LEARNING_MODEL_UPLOAD_DIR / \
                framework / str(model_path)

        elif model_type == ModelType.ProjectTrained:
            model_path = PROJECT_DIR / \
                get_directory_name(project_name) / 'training' / get_directory_name(
                    training_name) / 'exported_models' / str(model_path)

        # assert model_path.is_dir(), f"{str(model_path)} does not exists"

        if not new_model_flag and not model_path.is_dir():
            error_msg = f"{str(model_path)} does not exists"
            logger.error(error_msg)

        return model_path

    @staticmethod
    def query_project_model_path(model_id: int) -> Path:
        """Get path to Project Models

        Args:
            model_id (int): Model ID

        Returns:
            Path: Path-like object of the Project Model Path
        """
        query_model_project_training_SQL = """
            SELECT m.name AS "Name",
                (SELECT p.name AS "Project Name"
                    FROM public.project p
                            INNER JOIN project_training pt ON p.id = pt.project_id
                    WHERE m.training_id = pt.training_id),
                (SELECT t.name AS "Training Name" FROM public.training t WHERE m.training_id = t.id),
                (
                    SELECT f.name AS "Framework"
                    FROM public.framework f
                    WHERE f.id = m.framework_id
                ),
                m.model_path AS "Model Path"

            FROM public.models m
            WHERE m.id = %s;
                                    """
        query_model_project_training_vars = [model_id]
        query_result = db_fetchone(query_model_project_training_SQL,
                                   conn, query_model_project_training_vars)
        if query_result:

            project_model_path = PROJECT_DIR / \
                get_directory_name(query_result.Project_Name) / get_directory_name(
                    query_result.Training_Name) / 'exported_models' / query_result.Model_Path
            return project_model_path

    @staticmethod
    def generate_relative_model_path(model_name: str) -> str:
        """Generate model path relative to the parent folder
        - Utilised if 

        Args:
            model_name (str): Name of the model

        Returns:
            str: Relative model path
        """

        directory_name = get_directory_name(model_name)
        relative_model_path = f"./{directory_name}"

        return relative_model_path

    def insert_new_model(self, model_type: str = "User Deep Learning Model Upload") -> bool:
        """Create new row in `models` table

        Returns:
            bool: Return True if successful operation, otherwise False
        """        # create new row in Models table
        self.model_type = model_type

        insert_new_model_SQL = """
            INSERT INTO public.models (
                name
                , description
                , metrics
                , model_path
                , model_type_id
                , framework_id
                , deployment_id
                , training_id)
            VALUES (
                %s
                , %s
                , %s::jsonb
                , %s
                , (
                    SELECT
                        mt.id
                    FROM
                        public.model_type mt
                    WHERE
                        mt.name = %s) , (
                        SELECT
                            f.id
                        FROM
                            public.framework f
                        WHERE
                            f.name = %s) , (
                            SELECT
                                dt.id
                            FROM
                                public.deployment_type dt
                            WHERE
                                dt.name = %s) ,  %s)
                    RETURNING
                        id;

                                    """
        # self.model_path = self.get_pt_user_model_path(model_path=self.name,
        #                                               framework=self.framework,
        #                                               model_type=self.model_type,
        #                                               new_model_flag=True)

        # Generate relative model path
        self.model_path_relative = self.generate_relative_model_path(
            model_name=self.name)

        if self.model_input_size:
            self.metrics['metadata'] = self.model_input_size

        # SERIALISE Python Dictionary to JSON string
        if not isinstance(self.metrics, str):

            metrics_json = json.dumps(self.metrics)
        else:
            metrics_json = self.metrics

        insert_new_model_vars = [self.name, self.desc, metrics_json, str(self.model_path_relative),
                                 self.model_type, self.framework, self.deployment_type, self.training_id]

        try:
            query_return = db_fetchone(
                insert_new_model_SQL, conn, insert_new_model_vars).id

            assert isinstance(
                query_return, int), f"Model ID returned should be type int but type {type(query_return)} obtained ({query_return})"

            self.id = query_return
            return True

        except Exception as e:
            logger.error(
                f"{e}: Failed to create new row in Models table for {self.name}")
            return False

    def update_model_table(self, model_type: str = "User Deep Learning Model Upload"):
        self.model_type = model_type
        update_model_table_SQL = """
                UPDATE
                    public.models
                SET
                    name = %s
                    , description = %s
                    , metrics = %s::jsonb
                    , model_path = %s
                    , model_type_id = (
                        SELECT
                            mt.id
                        FROM
                            public.model_type mt
                        WHERE
                            mt.name = %s) , framework_id = (
                        SELECT
                            f.id
                        FROM
                            public.framework f
                        WHERE
                            f.name = %s) , deployment_id = (
                        SELECT
                            dt.id
                        FROM
                            public.deployment_type dt
                        WHERE
                            dt.name = %s) , training_id = %s
                WHERE
                    id = %s
                RETURNING
                    id;
                        
                        """
        # Generate relative model path
        self.model_path_relative = self.generate_relative_model_path(
            model_name=self.name)

        if self.model_input_size:
            self.metrics['metadata'] = self.model_input_size

        # SERIALISE Python Dictionary to JSON string
        if not isinstance(self.metrics, str):

            metrics_json = json.dumps(self.metrics)
        else:
            metrics_json = self.metrics

        update_model_table_vars = [self.name, self.desc, metrics_json, str(self.model_path_relative),
                                   self.model_type, self.framework, self.deployment_type, self.training_id, self.id]

        try:
            query_return = db_fetchone(
                update_model_table_SQL, conn, update_model_table_vars).id

            assert self.id == query_return, f'Updated wrong Model of ID {query_return}, which should be {self.id}'

            logger.info(
                f"Successfully updated New Training Model for {self.id} ")

            return True

        except Exception as e:
            logger.error(
                f"{e}: Failed to update Models table for {self.name}")
            return False

    def create_new_model_pipeline(self, label_map_string: str = None) -> bool:
        """Pipeline for new model upload creation
        - Get respective model path -> Extract uploaded model to the model path -> Generate labelmap file if needed -> Create new row at `models` table
        Args:
            label_map_string (str, optional): Labelmap of type str. Defaults to None.

        Returns:
            bool: True if successful process, False otherwise
        """

        with st.container():
            # get destination folder
            progress_bar = st.progress(0)
            self.model_type = "User Deep Learning Model Upload"

            self.model_path = self.get_pt_user_project_model_path(model_path=self.name,
                                                                  framework=self.framework,
                                                                  model_type=self.model_type,
                                                                  new_model_flag=True)
            logger.info(f"Model Path: {self.model_path}")
            # unpack
            progress_bar.progress(1 / 3)
            with st.spinner(text='Storing uploaded model'):
                save_uploaded_extract_files(dst=self.model_path,
                                            filename=self.file_upload.name,
                                            fileObj=self.file_upload)
            if label_map_string:
                # generate labelmap
                # move labelmap to dst
                Labels.generate_labelmap_file(labelmap_string=label_map_string,
                                              dst=self.model_path,
                                              framework=self.framework,
                                              deployment_type=self.deployment_type)

            # Create new row in DB
            progress_bar.progress(2 / 3)
            with st.spinner(text='Storing uploaded model'):
                self.insert_new_model(model_type=self.model_type)

            # Success msg
            progress_bar.progress(3 / 3)
            self.has_submitted = True
            st.success(f"Successfully uploaded new model: {self.name}")

        return True

    def create_new_project_model_pipeline(self,
                                          attached_model,
                                          project_name: str,
                                          training_name: str,
                                          training_id: int) -> bool:

        with st.container():

            # Ensure attached_model is type Model class
            assert isinstance(
                attached_model, Model), "attached_model must be type Model class"

            # Get metrics, model_type, framework and deployment_type from attached_model
            self.metrics = attached_model.metrics
            self.model_type = attached_model.model_type
            self.framework = attached_model.framework
            self.deployment_type = attached_model.deployment_type
            self.training_id = training_id

            # get destination folder
            self.model_type = 'Project Models'

            self.model_path = self.get_pt_user_project_model_path(model_path=self.name,
                                                                  framework=self.framework,
                                                                  project_name=project_name,
                                                                  training_name=training_name,
                                                                  model_type=self.model_type,
                                                                  new_model_flag=True)
            logger.info(f"Training Model Path: {str(self.model_path)}")

            # Create New Model Folder NOTE:KIV-> Create before training
            # logger.info(
            #     f"Creating Training Model folder at {str(self.model_path)}")
            # create_folder_if_not_exist(self.model_path)

            # Create new row in DB
            logger.info(f"Inserting Training Model into DB")
            self.insert_new_model(model_type=self.model_type)
            self.has_submitted = True

        return True

    def update_new_project_model_pipeline(self, attached_model,
                                          project_name: str,
                                          training_name: str) -> bool:

        with st.container():

            # Ensure attached_model is type Model class
            assert isinstance(
                attached_model, Model), "attached_model must be type Model class"

            # Get metrics, model_type, framework and deployment_type from attached_model
            self.metrics = attached_model.metrics
            self.model_type = attached_model.model_type
            self.framework = attached_model.framework
            self.deployment_type = attached_model.deployment_type

            # get destination folder
            self.model_type = 'Project Models'

            self.model_path = self.get_pt_user_project_model_path(model_path=self.name,
                                                                  framework=self.framework,
                                                                  project_name=project_name,
                                                                  training_name=training_name,
                                                                  model_type=self.model_type,
                                                                  new_model_flag=True)
            logger.info(f"Training Model Path: {str(self.model_path)}")

            # Create New Model Folder NOTE:KIV-> Create before training
            # logger.info(
            #     f"Creating Training Model folder at {str(self.model_path)}")
            # create_folder_if_not_exist(self.model_path)

            # Create new row in DB
            logger.info(f"Updating Training Model into DB")
            self.update_model_table(model_type=self.model_type)
            self.has_submitted = True

        return True

    @staticmethod
    @st.experimental_memo
    def get_pretrained_model_details(deployment_type: str, for_display: bool = False) -> pd.DataFrame:
        """Get the model details from the CSV files scraped from their websites,
        based on the `deployment_type`.

        Args:
            deployment_type (str): obtained from `Project.deployment_type` attribute
            for_display (bool, optional): True to drop irrelevant columns for displaying to the users. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with pretrained model details such as Model Name.
        """
        assert deployment_type in DEPLOYMENT_TYPE, "This should be from `Project.deployment_type`."

        if deployment_type == "Image Classification":
            # this df has columns: Model Name
            models_df = pd.read_csv(CLASSIF_MODELS_NAME_PATH)
        elif deployment_type == "Object Detection with Bounding Boxes":
            # this df has columns: Model Name, Speed (ms), COCO mAP, Outputs, model_links
            models_df = pd.read_csv(TFOD_MODELS_TABLE_PATH)
            if for_display:
                # `model_links` will be required for downloading the pretrained models for TFOD
                models_df.drop(columns='model_links', inplace=True)
        elif deployment_type == "Semantic Segmentation with Polygons":
            # this df has columns: model_func, Model Name, Reference, links
            # The `Reference` contains names of the authors
            # The `links` are just the paper's arXiv links
            models_df = pd.read_csv(SEGMENT_MODELS_TABLE_PATH)
            if for_display:
                # `model_func` are the functions required for using the `keras_unet_collection` library
                models_df.drop(columns='model_func', inplace=True)
        return models_df


class NewModel(BaseModel):
    def __init__(self, model_id: str = get_random_string(length=8)) -> None:
        super().__init__(model_id)
        self.has_submitted: bool = False
        self.deployment_type: str = False

# TODO TO be updated
    @staticmethod
    def reset_model_upload_page():
        """Reset session state attributes in user_model_upload pages,
        """
        new_model_attributes = ["model_upload", "labelmap",
                                "generate_labelmap_flag",
                                "model_upload_name",
                                "model_upload_desc",
                                "model_upload_deployment_type",
                                "model_upload_framework",
                                "model_upload_widget"]

        reset_page_attributes(new_model_attributes)

    @staticmethod
    def reset_models_page():
        """Reset session state attributes in user_model_upload pages,
        """
        new_model_attributes = ["training_model", "training_model_name",
                                ]

        reset_page_attributes(new_model_attributes)


class Model(BaseModel):
    def __init__(self, model_id: int = None, model_row: Dict = None) -> None:
        super().__init__(model_id)

        # ******************************IF GIVEN DATAFRAME ROW******************************
        if model_row:
            self.id: int = model_row['id']
            self.name: str = model_row['Name']
            self.desc: str = model_row['Description']
            self.metrics: Dict = model_row['Metrics']
            self.model_type: str = model_row['Model Type']
            self.framework: str = model_row['Framework']
            self.training_name: str = model_row['Training Name']
            self.updated_at: datetime = model_row['Date/Time']
            self.model_path_relative: str = model_row['Model Path']
            self.deployment_type: str = model_row['Deployment Type']

        # ****************************** IF GIVEN MODEL ID******************************
        elif model_id:
            assert (isinstance(
                model_id, int)), f"Model ID should be type int but type {type(model_id)} obtained ({model_id})"
            self.id = model_id
            # set up the remaining fields just like when `model_row` is being passed in
            self.query_all_fields()

        # ******************************Get model type constant******************************
        model_type_constant: IntEnum = self.get_model_type(
            self.model_type, string=False)

        # ****************************** Get respective model path******************************
        if (model_type_constant == ModelType.PreTrained) or (model_type_constant == ModelType.UserUpload):
            self.model_path = self.get_pt_user_project_model_path(model_path=self.model_path_relative,
                                                                  framework=self.framework,
                                                                  model_type=self.model_type)
        elif (model_type_constant == ModelType.ProjectTrained):
            self.model_path = self.get_project_model_path()

        # ******************************Get Model Input Size******************************
        # TODO: KIV
        # self.model_input_size = self.metrics.get('metadata').get('input_size')
        self.perf_metrics = self.get_perf_metrics()

    def __repr__(self):
        # return (f"Model(model_row={{'id': {self.id}, 'Name': {self.name}, "
        #         f"'Description': {self.desc}, "
        #         f"'Metrics': {self.metrics}, "
        #         f"'Model Type': {self.model_type}, "
        #         f"'Framework': {self.framework}, "
        #         f"'Training Name': {self.training_name}, "
        #         f"'Date/Time': {self.updated_at}, "
        #         f"'Model Path': {self.model_path_relative}, "
        #         f"'Deployment Type': {self.deployment_type}"
        #         "})"
        #         )
        return "<{klass} {attrs}>".format(
            klass=self.__class__.__name__,
            attrs=" ".join("{}={!r}".format(k, v)
                           for k, v in self.__dict__.items()),
        )

    def query_all_fields(self) -> NamedTuple:

        query_all_fields_SQL = """
                SELECT
                    m.id AS "ID"
                    , m.name AS "Name"
                    , (
                        SELECT
                            f.name AS "Framework"
                        FROM
                            public.framework f
                        WHERE
                            f.id = m.framework_id) , (
                        SELECT
                            mt.name AS "Model Type"
                        FROM
                            public.model_type mt
                        WHERE
                            mt.id = m.model_type_id) , (
                        SELECT
                            dt.name AS "Deployment Type"
                        FROM
                            public.deployment_type dt
                        WHERE
                            dt.id = m.deployment_id) , (
                        /* Replace NULL with '-' */
                        SELECT
                            CASE WHEN m.training_id IS NULL THEN
                                '-'
                            ELSE
                                (
                                    SELECT
                                        t.name
                                    FROM
                                        public.training t
                                    WHERE
                                        t.id = m.training_id)
                            END AS "Training Name")
                    , m.updated_at AS "Date/Time"
                    , m.description AS "Description"
                    , m.metrics AS "Metrics"
                    , m.model_path AS "Model Path"
                FROM
                    public.models m
                WHERE
                    m.id = %s;
                """
        query_all_fields_vars = [self.id]

        model_field = db_fetchone(query_all_fields_SQL,
                                  conn,
                                  query_all_fields_vars)

        if model_field:

            self.id, self.name, self.framework, self.model_type, self.deployment_type,\
                self.training_name, self.updated_at, self.desc, self.metrics,\
                self.model_path_relative = model_field
        else:
            logger.error(
                f"Model with ID {self.id} does not exists in the Database!!!")

            model_field = None

        return model_field

    @staticmethod
    def query_model_table(for_data_table: bool = False, return_dict: bool = False, deployment_type: Union[str, IntEnum] = None) -> namedtuple:
        """Wrapper function to query model table

        Args:
            for_data_table (bool, optional): True if query for data table. Defaults to False.
            return_dict (bool, optional): True if query results of type Dict. Defaults to False.
            deployment_type (IntEnum, optional): Deployment type. Defaults to None.

        Returns:
            namedtuple: [description]
        """
        if deployment_type:
            models, column_names = query_model_ref_deployment_type(deployment_type=deployment_type,
                                                                   for_data_table=for_data_table,
                                                                   return_dict=return_dict)

        else:
            models, column_names = query_all_models(
                for_data_table=for_data_table, return_dict=return_dict)

        return models, column_names

    @staticmethod
    @st.cache
    def get_framework_list() -> List[namedtuple]:
        """Get list of Deep Learning frameworks from Database

        Returns:
            List[namedtuple]: List of framework in namedtuple (ID, Name)
        """
        get_framework_list_SQL = """
            SELECT
             
                name as "Name"
            FROM
                public.framework;
                    """
        framework_list = db_fetchall(get_framework_list_SQL, conn)
        return framework_list

    @staticmethod
    def create_models_dataframe(models: Union[List[namedtuple], List[dict]],
                                column_names: List = None, sort_col: str = None
                                ) -> pd.DataFrame:
        """Generate Pandas DataFrame to store Models query

        Args:
            models (Union[List[namedtuple], List[dict]]): Models query from 'query_model_table()'
            column_names (List, optional): Names of columns. Defaults to None.
            sort_col (str, optional): Sort value. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame of Models query
        """
        df = create_dataframe(models, column_names,
                              date_time_format=True, sort_by=sort_col, asc=True)

        df['Date/Time'] = df['Date/Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        return df

    @staticmethod
    @dataframe2dict(orient='index')
    def filtered_models_dataframe(models: Union[List[namedtuple], List[dict]],
                                  dataframe_col: str, filter_value: Union[str, int],
                                  column_names: List = None, sort_col: str = None
                                  ) -> pd.DataFrame:
        """Get a List of filtered Models Dict using pandas.DataFrame.loc[]

        Args:
            models (Union[List[namedtuple], List[dict]]): models query from 'query_models_table'
            dataframe_col (str): DataFrame column to be filtered
            filter_value (Union[str, int]): Filter attribute

        Returns:
            List[Dict]: Filtered DataFrame
        """

        models_df = Model.create_models_dataframe(
            models, column_names, sort_col=sort_col)
        filtered_models_df = models_df.loc[models_df[dataframe_col]
                                           == filter_value]

        return filtered_models_df

    def get_project_model_path(self):
        self.model_path = BaseModel.query_project_model_path(self.id)
        return self.model_path

    def get_labelmap_path(self):
        model_path = self.get_model_path()
        if model_path:
            labelmap_path = model_path / 'labelmap.pbtxt'
            self.labelmap_path = labelmap_path

            return self.labelmap_path

    def get_model_row(model_id: int, model_df: pd.DataFrame) -> Dict:
        """Get selected model row

        Args:
            model_id (int): Model ID
            model_df (pd.DataFrame): DataFrame for models

        Returns:
            Dict: Data row from models DataFrame
        """
        logger.info(f"Obtaining data row from model_df......")

        model_row = get_dataframe_row(model_id, model_df)

        logger.info(f"Currently serving data:{model_row['Name']}")

        return model_row

    def get_perf_metrics(self):
        perf_metrics = []
        deployment_type = Deployment.get_deployment_type(
            self.deployment_type, string=False)
        if self.metrics.get('evaluation'):
            for name, values in self.metrics.get('evaluation').items():
                if name in EVALUATION_TAGS[deployment_type]:
                    for i in values:
                        i['metrics'] = name
                        perf_metrics.append(i)
        return perf_metrics

    @staticmethod
    def create_perf_metrics_table(data: List) -> pd.DataFrame:
        df_metrics = pd.DataFrame(
            data, columns=['metrics', 'name', 'value', 'unit'])
        df_metrics['value'].map(
            "{:.2f}".format)  # Only show 2 DP for DataFrame
        df_metrics = df_metrics.set_index(['metrics'])

        return df_metrics


# ********************************** DEPRECATED ********************************


class PreTrainedModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.pt_model_list, self.pt_model_column_names = self.query_PT_table()

    # DEPRECATED?
    @st.cache
    def query_PT_table(self) -> NamedTuple:
        query_PT_table_SQL = """
            SELECT
                pt.id AS "ID",
                pt.name AS "Name",
                f.name AS "Framework",
                dt.name AS "Deployment Type",
                pt.model_path AS "Model Path"
            FROM
                public.pre_trained_models pt
                LEFT JOIN public.framework f ON f.id = pt.framework_id
                LEFT JOIN public.deployment_type dt ON dt.id = pt.deployment_id;"""
        PT_model_list, column_names = db_fetchall(
            query_PT_table_SQL, conn, fetch_col_name=True)
        return PT_model_list, column_names


@st.cache(ttl=60)
def query_all_models(for_data_table: bool = False, return_dict: bool = False):

    ID_string = "id" if for_data_table else "ID"

    query_all_model_SQL = f"""
            SELECT
                m.id AS \"{ID_string}\",
                m.name AS "Name",
                (
                    SELECT
                        f.name AS "Framework"
                    FROM
                        public.framework f
                    WHERE
                        f.id = m.framework_id
                ),
                (
                    SELECT
                        mt.name AS "Model Type"
                    FROM
                        public.model_type mt
                    WHERE
                        mt.id = m.model_type_id
                ),
                (
                    SELECT
                        dt.name AS "Deployment Type"
                    FROM
                        public.deployment_type dt
                    WHERE
                        dt.id = m.deployment_id
                ),
                (
                    /* Replace NULL with '-' */
                    SELECT
                        CASE
                            WHEN m.training_id IS NULL THEN '-'
                            ELSE (
                                SELECT
                                    t.name
                                FROM
                                    public.training t
                                WHERE
                                    t.id = m.training_id
                            )
                        END AS "Training Name"
                ),
                m.description AS "Description",
                m.metrics AS "Metrics",
                m.model_path AS "Model Path"
            FROM
                public.models m
            ORDER BY
                ID ASC;
                    """

    models, column_names = db_fetchall(
        query_all_model_SQL, conn, fetch_col_name=True, return_dict=return_dict)

    logger.info(f"Querying all models......")

    models_tmp = []

    if models:
        models_tmp = datetime_formatter(models, return_dict=return_dict)

    else:
        models_tmp = []

    return models_tmp, column_names


@st.cache(ttl=60)
def query_model_ref_deployment_type(deployment_type: Union[str, IntEnum] = None, for_data_table: bool = False, return_dict: bool = False):
    """Query rows of models filtered by Deployment Type from 'models' table

    Args:
        deployment_type (str, optional): [description]. Defaults to None.
        for_data_table (bool, optional): [description]. Defaults to False.
        return_dict (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    deployment_type = Deployment.get_deployment_type(
        deployment_type=deployment_type, string=True)
    ID_string = "id" if for_data_table else "ID"

    model_dt_SQL = """
        SELECT
            m.id AS \"{ID_string}\",
            m.name AS "Name",
            (
                SELECT
                    f.name AS "Framework"
                FROM
                    public.framework f
                WHERE
                    f.id = m.framework_id
            ),
            (
                SELECT
                    mt.name AS "Model Type"
                FROM
                    public.model_type mt
                WHERE
                    mt.id = m.model_type_id
            ),
            (
                /* Replace NULL with '-' */
                SELECT
                    CASE
                        WHEN m.training_id IS NULL THEN '-'
                        ELSE (
                            SELECT
                                t.name
                            FROM
                                public.training t
                            WHERE
                                t.id = m.training_id
                        )
                    END AS "Training Name"
            ),
                   m.updated_at  AS "Date/Time",
            m.description AS "Description",
            m.metrics AS "Metrics",
            m.model_path AS "Model Path",
            dt.name AS "Deployment Type"
        FROM
            public.models m
            INNER JOIN public.deployment_type dt ON dt.name = %s
        ORDER BY
            m.id ASC;        
                """.format(ID_string=ID_string)

    model_dt_vars = [deployment_type]

    models, column_names = db_fetchall(
        model_dt_SQL, conn, model_dt_vars, fetch_col_name=True, return_dict=return_dict)

    logger.info(f"Querying models filtered by Deployment Type from database....")

    models_tmp = []

    if models:
        models_tmp = datetime_formatter(models, return_dict)

    else:
        models_tmp = []

    return models_tmp, column_names
