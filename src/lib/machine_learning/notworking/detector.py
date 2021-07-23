"""
Title: Object Detection Class
Date: 23/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
Description: Object detection
"""
import queue
import sys
import os
from pathlib import Path
from typing import NamedTuple, List, Dict, Union
from queue import Queue
import numpy as np

import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from google.protobuf.pyext._message import RepeatedCompositeContainer

import streamlit as st
from streamlit import session_state as SessionState
# NEW
from streamlit_webrtc import (
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
import av
# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC = Path(__file__).resolve().parents[3]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

os.chdir(str(PROJECT_ROOT))
print(PROJECT_ROOT)
# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
from module.frame_overlay import draw_overlay
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Setup WebRTC >>>>
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)
# <<<< Setup WebRTC <<<<

@st.cache(hash_funcs={RepeatedCompositeContainer: id})
def load_model(configs):
    log_info(f"Loading model......")
    model_config = configs['model']
    detection_model = model_builder.build(
        model_config=model_config, is_training=False)
    log_info(f"Loaded model!")
    return detection_model



class Model:
    def __init__(self, models_dir) -> None:
        self.models_dir: Path = Path(models_dir)
        self.path_to_ckpt: Path = self.models_dir / 'checkpoint'
        self.path_to_cfg: Path = self.models_dir / 'pipeline.config'
        self.path_to_label: Path = self.models_dir / 'labelmap.pbtxt'
        log_info(f"Loading configuration from <{str(self.path_to_cfg)}>")
        self._configs = config_util.get_configs_from_pipeline_file(
            str(self.path_to_cfg))
        log_info(f"Loaded configuration from <{str(self.path_to_cfg)}>!!!")
        self.load_model()
        # restore checkpoint
        log_info("Restoring Checkpoint......")
        self.ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        _restore_ckpt = self.path_to_ckpt / 'ckpt-0'
        self.ckpt.restore(str(_restore_ckpt)).expect_partial()
        log_info("Checkpoint restored!!!")

        log_info("Loading Index....")
        self.category_index = label_map_util.create_category_index_from_labelmap(str(self.path_to_label),
                                                                                 use_display_name=True)
        log_info(
            " ************************** Loaded Index ************************** ")

    @st.cache(hash_funcs={RepeatedCompositeContainer: id})
    def load_model(self):
        log_info(f"Loading model......")
        self._model_config = self._configs['model']
        self.detection_model = model_builder.build(
            model_config=self._model_config, is_training=False)
        log_info(f"Loaded model!")




st.write("Hi")
