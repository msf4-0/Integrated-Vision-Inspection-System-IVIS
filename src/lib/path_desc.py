"""
Title: Path Description
Date: 5/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
Description:
- Show Root of Project

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

import os
import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp
from contextlib import contextmanager
from appdirs import user_config_dir, user_data_dir

from core.utils.log import logger

_CURR_FILEPATH = Path(__file__).resolve()

_DIR_APP_NAME = "integrated-vision-inspection-system"
_DIR_AUTHOR_NAME = "SHRDC"

# REFERENCED LS


def get_config_dir():
    config_dir = user_config_dir(
        appname=_DIR_APP_NAME,
        appauthor=_DIR_AUTHOR_NAME)
    os.makedirs(config_dir, exist_ok=True)
    return config_dir


def get_data_dir():
    data_dir = user_data_dir(
        appname=_DIR_APP_NAME,
        appauthor=_DIR_AUTHOR_NAME)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


@contextmanager
def get_temp_dir() -> str:
    dirpath = mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


# ./image_labelling_shrdc
PROJECT_ROOT = _CURR_FILEPATH.parents[2]
SECRETS_PATH = PROJECT_ROOT / ".streamlit" / "secrets.toml"

# DATA_DIR = Path.home() / '.local/share/integrated-vision-inspection-system/app_media'
BASE_DATA_DIR = Path(get_data_dir())
MEDIA_ROOT = BASE_DATA_DIR / 'app_media'
DATABASE_DIR = MEDIA_ROOT / 'data'
if not DATABASE_DIR.exists():
    os.makedirs(DATABASE_DIR)
DATASET_DIR = MEDIA_ROOT / 'dataset'
if not DATASET_DIR.exists():
    os.makedirs(DATASET_DIR)
PROJECT_DIR = MEDIA_ROOT / 'project'
if not PROJECT_DIR.exists():
    os.makedirs(PROJECT_DIR)
PRE_TRAINED_MODEL_DIR = MEDIA_ROOT / 'models' / 'pre-trained-models'
if not PRE_TRAINED_MODEL_DIR.exists():
    os.makedirs(PRE_TRAINED_MODEL_DIR)
USER_DEEP_LEARNING_MODEL_UPLOAD_DIR = MEDIA_ROOT / \
    'models' / 'user-deep-learning-model-upload'
if not USER_DEEP_LEARNING_MODEL_UPLOAD_DIR.exists():
    os.makedirs(USER_DEEP_LEARNING_MODEL_UPLOAD_DIR)
# PROJECT_MODELS=PROJECT_DIR/<PROJECT-NAME>/<TRAINING-NAME>/'exported-models'/<MODEL-NAME>
# named temporary directory
TEMP_DIR = BASE_DATA_DIR / 'temp'
CAPTURED_IMAGES_DIR = MEDIA_ROOT / 'captured_images'
if not CAPTURED_IMAGES_DIR.exists():
    os.makedirs(CAPTURED_IMAGES_DIR)

# Pretrained model details
# assuming this folder is in "utils/resources/" directory
# NOTE: this foldername is also being used in Dockerfile and .dockerignore
PRETRAINED_MODEL_TABLES_DIR = _CURR_FILEPATH.parents[2] / \
    'resources' / 'pretrained_model_tables'
if not PRETRAINED_MODEL_TABLES_DIR.exists():
    os.makedirs(PRETRAINED_MODEL_TABLES_DIR)
# this table has columns: Model Name
TFOD_MODELS_TABLE_PATH = PRETRAINED_MODEL_TABLES_DIR / 'tfod_pretrained_models.csv'
# Keras image classification pretrained model names from
# https://www.tensorflow.org/api_docs/python/tf/keras/applications
# this table has columns: Model Name
CLASSIF_MODELS_NAME_PATH = PRETRAINED_MODEL_TABLES_DIR / \
    'classif_pretrained_models.csv'
# this table has columns: model_func, Model Name, Reference, links
SEGMENT_MODELS_TABLE_PATH = PRETRAINED_MODEL_TABLES_DIR / \
    'segment_pretrained_models.csv'

# folder to store the code cloned for TensorFlow Object Detection (TFOD)
# from https://github.com/tensorflow/models
TFOD_DIR = _CURR_FILEPATH.parent / "TFOD" / "models"

MQTT_CONFIG_PATH = PROJECT_ROOT / 'src/lib/deployment/mqtt_config.yml'


def chdir_root():
    os.chdir(str(PROJECT_ROOT))
    logger.debug(f"Current working directory: {str(PROJECT_ROOT)} ")
    logger.debug(f"Data Directory set to \'{BASE_DATA_DIR}\'")


def add_path(node: str, parent_node: int = 0) -> None:
    SRC = _CURR_FILEPATH.parents[parent_node]  # ROOT folder -> ./src
    if node is not None:
        PATH = SRC / node  # ./PROJECT_ROOT/src/lib
    else:
        PATH = SRC
    # CHECK if PATH exists
    try:
        PATH.resolve(strict=True)
    except FileNotFoundError:
        logger.error(f"Path {PATH} does not exist")
    else:
        if str(PATH) not in sys.path:
            sys.path.insert(0, str(PATH))  # ./lib
        else:
            logger.debug(
                f"\'{PATH.relative_to(PROJECT_ROOT.parent)} \'added into Python PATH")
            pass
