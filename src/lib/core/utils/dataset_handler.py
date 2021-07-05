"""
Title: Dataset Handler
Date: 30/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import os
from pathlib import Path, PurePath
import re
from shutil import Error, copyfile
import argparse
import math
import random
import logging
from glob import glob, iglob
import streamlit as st
import psycopg2
import sys

#--------------------Logger-------------------------#
FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
DATEFMT = '%d-%b-%y %H:%M:%S'

# logging.basicConfig(filename='test.log',filemode='w',format=FORMAT, level=logging.DEBUG)
logging.basicConfig(format=FORMAT, level=logging.INFO,
                    stream=sys.stdout, datefmt=DATEFMT)

log = logging.getLogger()

#----------------------------------------------------#
# ------------------TEMP
conn = psycopg2.connect(
    "host=localhost port=5432 dbname=eye user=shrdc password=shrdc")
layout = 'centered'
# ------------------TEMP


HOME = Path.home()


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
            log.info(data_path)

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
    output_dir = Path(project_dir, "image")
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
            log.error(error_message)
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
        log.error(error_message)
        st.error(error_message)

    return str(data_path), str(output_dir)


def main():
    dataset_partition()


if __name__ == '__main__':
    main()
