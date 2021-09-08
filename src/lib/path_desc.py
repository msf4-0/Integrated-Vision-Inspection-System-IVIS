"""
Title: Path Description
Date: 5/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
Description:
- Show Root of Project
"""

import os
import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp, mkstemp
from contextlib import contextmanager
from appdirs import user_config_dir, user_data_dir

from core.utils.log import log_error, log_info

_DIR_APP_NAME = "integrated-vision-inspection-system"

# REFERENCED LS


def get_config_dir():
    config_dir = user_config_dir(appname=_DIR_APP_NAME)
    os.makedirs(config_dir, exist_ok=True)
    return config_dir


def get_data_dir():
    data_dir = user_data_dir(appname=_DIR_APP_NAME)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


@contextmanager
def get_temp_dir():
    dirpath = mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


# ./image_labelling_shrdc
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# DATA_DIR = Path.home() / '.local/share/integrated-vision-inspection-system/app_media'
BASE_DATA_DIR = Path(get_data_dir())
MEDIA_ROOT = BASE_DATA_DIR / 'app_media'
DATASET_DIR = MEDIA_ROOT / 'dataset'
PROJECT_DIR = MEDIA_ROOT / 'project'
PRE_TRAINED_MODEL_DIR = MEDIA_ROOT / 'pre-trained-models'
USER_DEEP_LEARNING_MODEL_UPLOAD_DIR = MEDIA_ROOT / \
    'user-deep-learning-model-upload'
# PROJECT_MODELS=PROJECT_DIR/<PROJECT-NAME>/<TRAINING-NAME>/'exported-models'/<MODEL-NAME>


def chdir_root():
    os.chdir(str(PROJECT_ROOT))
    log_info(f"Current working directory: {str(PROJECT_ROOT)} ")
    log_info(f"Data Directory set to \'{BASE_DATA_DIR}\'")


def add_path(node: str, parent_node: int = 0) -> None:
    SRC = Path(__file__).resolve().parents[parent_node]  # ROOT folder -> ./src
    if node is not None:
        PATH = SRC / node  # ./PROJECT_ROOT/src/lib
    else:
        PATH = SRC
    # CHECK if PATH exists
    try:
        PATH.resolve(strict=True)
    except FileNotFoundError:
        log_error(f"Path {PATH} does not exist")
    else:
        if str(PATH) not in sys.path:
            sys.path.insert(0, str(PATH))  # ./lib
        else:
            log_info(
                f"\'{PATH.relative_to(PROJECT_ROOT.parent)} \'added into Python PATH")
            pass
