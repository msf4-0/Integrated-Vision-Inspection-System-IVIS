"""
Title: Path Description
Date: 5/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
Description:
- Show Root of Project
"""

import sys
import os
from pathlib import Path
from core.utils.log import log_info, log_error
from core.utils.file_handler import get_config_dir, get_data_dir

# ./image_labelling_shrdc
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# DATA_DIR = Path.home() / '.local/share/integrated-vision-inspection-system/app_media'
BASE_DATA_DIR = Path(get_data_dir())
MEDIA_ROOT = BASE_DATA_DIR / 'app_media'
DATASET_DIR = MEDIA_ROOT / 'dataset'
PROJECT_DIR = MEDIA_ROOT / 'project'
PRE_TRAINED_MODEL_DIR = MEDIA_ROOT / 'pre-trained-models'
USER_DEEP_LEARNING_MODEL_UPLOAD = MEDIA_ROOT / 'user-deep-learning-model-upload'

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
