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

# ! DEPRECATED: RENAMED folder path to a shorter name
_OLD_DIR_APP_NAME = "integrated-vision-inspection-system"
_DIR_APP_NAME = "IVIS"
_DIR_AUTHOR_NAME = "SHRDC"


def rename_dir_for_backward_compatibility(old_dir: str, new_dir: str):
    if os.path.exists(old_dir):
        logger.info("Renaming old data directory to new name, "
                    f"from '{old_dir}' to '{new_dir}'")
        os.rename(old_dir, new_dir)


# REFERENCED LS


def get_config_dir():
    config_dir = user_config_dir(
        appname=_DIR_APP_NAME,
        appauthor=_DIR_AUTHOR_NAME)
    old_config_dir = user_config_dir(
        appname=_OLD_DIR_APP_NAME,
        appauthor=_DIR_AUTHOR_NAME)

    rename_dir_for_backward_compatibility(
        old_config_dir, config_dir)

    # then only create if really not exists
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    return config_dir


def get_data_dir():
    data_dir = user_data_dir(
        appname=_DIR_APP_NAME,
        appauthor=_DIR_AUTHOR_NAME)
    old_data_dir  = user_data_dir(
        appname=_OLD_DIR_APP_NAME,
        appauthor=_DIR_AUTHOR_NAME)

    rename_dir_for_backward_compatibility(
        old_data_dir, data_dir)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir


@contextmanager
def get_temp_dir() -> str:
    dirpath = mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


# ./image_labelling_shrdc
PROJECT_ROOT = _CURR_FILEPATH.parents[2]
SECRETS_PATH = PROJECT_ROOT / ".streamlit" / "secrets.toml"

# DATA_DIR = Path.home() / '.local/share/IVIS/app_media'
BASE_DATA_DIR = Path(get_data_dir())
MEDIA_ROOT = BASE_DATA_DIR / 'app_media'
DATABASE_DIR = MEDIA_ROOT / 'data'
DATASET_DIR = MEDIA_ROOT / 'dataset'
PROJECT_DIR = MEDIA_ROOT / 'project'
PRE_TRAINED_MODEL_DIR = MEDIA_ROOT / 'models' / 'pre-trained-models'
USER_DEEP_LEARNING_MODEL_UPLOAD_DIR = MEDIA_ROOT / \
    'models' / 'user-deep-learning-model-upload'
# PROJECT_MODELS=PROJECT_DIR/<PROJECT-NAME>/<TRAINING-NAME>/'exported-models'/<MODEL-NAME>
# named temporary directory
TEMP_DIR = BASE_DATA_DIR / 'temp'
CAPTURED_IMAGES_DIR = MEDIA_ROOT / 'captured_images'

# Pretrained model details
# assuming this folder is in "utils/resources/" directory
PRETRAINED_MODEL_TABLES_DIR = _CURR_FILEPATH.parents[2] / \
    'resources' / 'pretrained_model_tables'
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

dirs_to_create = (DATABASE_DIR, DATASET_DIR, PROJECT_DIR, PRE_TRAINED_MODEL_DIR,
                  USER_DEEP_LEARNING_MODEL_UPLOAD_DIR, CAPTURED_IMAGES_DIR,
                  PRETRAINED_MODEL_TABLES_DIR)
for d in dirs_to_create:
    if not d.exists():
        os.makedirs(d)

paths_to_del = (TEMP_DIR,)
for p in paths_to_del:
    if p.exists():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


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
