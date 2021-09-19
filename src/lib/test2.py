"""
TEST
Description: Some file for random testing

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
from time import sleep
import cv2
import os
import numpy as np
import pandas as pd
import datetime
import psycopg2
from io import BytesIO
from shutil import make_archive
import streamlit as st
from PIL import Image
from streamlit import session_state as session_state
from streamlit.report_thread import add_report_ctx

SRC = Path(__file__).resolve().parents[1]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass
from training.labelmap_generator import labelmap_generator
from core.utils.log import log_info

labelmap_generator(framework='TensorFlow',
                   deployment_type='Object Detection with Bounding Boxes')


# Store in dictionary
