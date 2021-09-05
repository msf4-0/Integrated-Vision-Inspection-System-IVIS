"""
Title: Dataset Handler
Date: 30/6/2021
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

import os
from pathlib import Path, PurePath
import sys
import re
from shutil import Error, copyfile
import argparse
import math
import random
import logging
from glob import glob, iglob
import streamlit as st
import psycopg2
from io import BytesIO
from PIL import Image
from mimetypes import guess_type
from base64 import b64encode
import numpy as np
import cv2
from typing import Union, List, Dict, Optional
from enum import IntEnum


# >>>> User-defined Modules >>>>
SRC = Path(__file__).resolve().parents[3]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
from data_manager.database_manager import init_connection


conn = init_connection(**st.secrets["postgres"])


def query_dataset_dir(project_id, conn=conn):
    """Query database to obtain path to dt

    Args:
        project_id ([type]): [description]
        conn ([type], optional): [description]. Defaults to conn.

    Returns:
        [type]: [description]
    """

    data_path = []
    SQL_query_data_dir = """
                        SELECT data_path
                        FROM project
                        WHERE project_id = %s AND skip = false
                        ORDER BY task_id
                        """
    with conn:
        with conn.cursor() as cur:
            cur.execute(SQL_query_data_dir, project_id)

            conn.commit()
            data_path = cur.fetchall()
            log_info(data_path)

    return data_path


def iterate_dir(data_path, output_dir, num_data_partition, xml_flag=False, conn=conn):
    """[summary]

    Args:
        data_path ([type]): [description]
        output_dir ([type]): [description]
        num_data_partition ([type]): [description]
        xml_flag (bool, optional): [description]. Defaults to False.
        conn ([type], optional): [description]. Defaults to conn.

    Returns:
        [type]: [description]
    """
    # source = source.replace('\\', '/')
    # dest = dest.replace('\\', '/')
    # train_dir = path.join(dest, 'train')
    # test_dir = path.join(dest, 'test')

    # if not path.exists(train_dir):
    #     makedirs(train_dir)
    # if not path.exists(test_dir):
    #     makedirs(test_dir)

    # images = [f for f in listdir(source)
    #           if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.jpg|.jpeg|.png)$', f)]

    for i in range(num_data_partition):
        idx = random.randint(0, len(data_path) - 1)
        # get filename of data in Path object
        data = data_path.pop(idx)  # remove from list
        filename = Path(data).name
        copyfile(data,
                 Path(output_dir, filename))
        if xml_flag:
            xml_path = data.stem + '.xml'
            copyfile(xml_path, Path(output_dir, xml_path.name))
        # data_path.remove(data_path[idx])

    # for filename in images:
    #     copyfile(path.join(data_path, filename),
    #              path.join(train_dir, filename))
    #     if xml_flag:
    #         xml_filename = path.splitext(filename)[0] + '.xml'
    #         copyfile(path.join(data_path, xml_filename),
    #                  path.join(train_dir, xml_filename))
    return data_path
# need to find out project_id


def dataset_partition(project_id, data_path, project_dir=Path.cwd(), a=0.9, b=0.1, c=None, test_partition_flag=False, xml_flag=False, conn=conn):
    """[summary]

    Args:
        project_id ([type]): [description]
        data_path ([type]): [description]
        project_dir ([type], optional): [description]. Defaults to Path.cwd().
        a (float, optional): [description]. Defaults to 0.9.
        b (float, optional): [description]. Defaults to 0.1.
        c ([type], optional): [description]. Defaults to None.
        test_partition_flag (bool, optional): [description]. Defaults to False.
        xml_flag (bool, optional): [description]. Defaults to False.
        conn ([type], optional): [description]. Defaults to conn.

    Returns:
        [type]: [description]
    """

    # instantiate Path objects for dataset and output directory
    data_path = Path(data_path)  # KIV
    project_dir = Path(project_dir)
    output_dir = Path(project_dir, "dataset")
    train_img_dir = Path(output_dir, "train")
    eval_img_dir = Path(output_dir, "evaluation")
    test_img_dir = Path(output_dir, "test")

    if not data_path.is_dir():
        try:
            search_dir_list = []
            for search_dir in iglob(str(Path(project_dir, "**", data_path.parts[-1])), recursive=True):
                search_dir_list.append(search_dir)
            assert len(search_dir_list) > 0, "Dataset directory is missing"
        except OSError as e:
            error_message = f"{e}Missing dataset directory"
            log_error(error_message)
            st.error(error_message)

    if output_dir is None:
        output_dir = data_path
    if not train_img_dir.exists():  # if train dataset directory does not exist

        train_img_dir.mkdir(parents=True)  # make parent directory if not exist
        if not eval_img_dir.exists():  # if evaluation dataset directory does not exist
            eval_img_dir.mkdir(parents=True)

        if test_partition_flag:
            # if require partition dataset for testing
            # generate test dataset if test_partition_flag is True
            test_img_dir.mkdir(parents=True)

    # Query database to obtain list of datasets
    # get project ID from SessionState
    data_path = query_dataset_dir(project_id, conn)

    num_data = len(data_path)
    num_eval_data = math.ceil(b * num_data)
    data_path = iterate_dir(data_path, eval_img_dir, num_eval_data, xml_flag)

    if test_partition_flag:
        num_test_data = math.ceil(c * num_data)
        data_path = iterate_dir(data_path, test_img_dir,
                                num_test_data, xml_flag)
    else:
        num_test_data = 0

    # get train partition
    train_data_path = data_path

    try:
        assert len(train_data_path) > 0
    except Error as e:
        error_message = f"{e}Train dataset empty"
        log_error(error_message)
        st.error(error_message)

    return str(data_path), str(output_dir)

# PIL


@st.cache
def data_url_encoder_PIL(image: Image):
    """Load Image and generate Data URL in base64 bytes

    Args:
        image (bytes-like): BytesIO object

    Returns:
        bytes: UTF-8 encoded base64 bytes
    """
    img_byte = BytesIO()
    image_name = Path(image.filename).name  # use Path().name
    log_info(f"Encoding image into bytes: {str(image_name)}")
    image.save(img_byte, format=image.format)
    log_info("Done enconding into bytes")

    log_info("Start B64 Encoding")
    bb = img_byte.getvalue()
    b64code = b64encode(bb).decode('utf-8')
    log_info("Done B64 encoding")

    mime = guess_type(image.filename)[0]
    log_info(f"{image_name} ; {mime}")
    data_url = f"data:{mime};base64{b64code}"
    log_info("Data url generated")

    return data_url


@st.cache(show_spinner=True)
def load_image_PIL(image_path: Path) -> str:

    log_info("Loading Image")
    if image_path.is_file():
        try:
            img = Image.open(image_path)
        except Exception as e:
            log_error(f"{e}: Failed to load image")
            img = None

    else:
        img = None
        exception = FileNotFoundError
        st.error(f"{exception} Image does not exists in dataset")
        raise FileNotFoundError("Image does not exists in dataset")

    return img

# OpenCV


@st.cache
def data_url_encoder_cv2(image: np.ndarray, image_name: str):
    """Load Image and generate Data URL in base64 bytes

    Args:
        image (bytes-like): BytesIO object

    Returns:
        bytes: UTF-8 encoded base64 bytes
    """

    log_info(f"Encoding image into bytes: {str(image_name)}")
    extension = Path(image_name).suffix
    _, buffer = cv2.imencode(extension, image)
    log_info("Done enconding into bytes")

    log_info("Start B64 Encoding")

    b64code = b64encode(buffer).decode('utf-8')
    log_info("Done B64 encoding")

    mime = guess_type(image_name)[0]
    log_info(f"{image_name} ; {mime}")
    data_url = f"data:{mime};base64,{b64code}"
    log_info("Data url generated")

    return data_url


@st.cache
def load_image_cv2(image_path: str) -> str:

    log_info("Loading Image")

    img = cv2.imread(image_path)
    return img



def get_image_size(image: Union[np.ndarray, Image.Image]):
    """get dimension of image

    Args:
        image_path (str): path to image or byte_like object

    Returns:
        tuple: original_width and original_height
    """
    if image:
        if isinstance(image, np.ndarray):
            original_width, original_height = image.shape[1], image.shape[0]
        elif isinstance(image, Image.Image):

            original_width, original_height = image.size

    else:
        original_width, original_height = None, None

    return original_width, original_height


def main():
    pass


if __name__ == '__main__':
    main()
