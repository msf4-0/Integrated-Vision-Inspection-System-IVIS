"""
General Parser:
1. JSON
2. YAML
Date: 21/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)

"""
from pathlib import Path
from glob import glob, iglob
import json
import io
import yaml
import os
import shutil
import streamlit as st
import logging

#--------------------Logger-------------------------#
FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
DATEFMT = '%d-%b-%y %H:%M:%S'

# logging.basicConfig(filename='test.log',filemode='w',format=FORMAT, level=logging.DEBUG)
logging.basicConfig(format=FORMAT, level=logging.INFO,
                    stream=sys.stdout, datefmt=DATEFMT)

log = logging.getLogger()

#----------------------------------------------------#

#-----------JSON PARSER-------------------#


#-----------YAML PARSER-------------------#

#-----------FILE HANDLER------------------#
def file_open(path):
    """general file opener"""
    with open(path, mode='r') as file:
        file = file.read()
    return file


def multi_file_open(path):
    """general multi-file opener"""

    file_list = []
    with open(path, mode='r') as files:
        file_list = files
    return file_list


def file_search(path=str(Path.home())):
    """
    File Search
    - Recursive true to search sub-directories with " ** "
    - Use wild-card for non-specific files ('*.extension')
    - CWD/*/*
    """
    file_list = []
    file_list = glob(pathname=path)
    # for file in glob(pathname=path):
    #     file_list.append(file)

    return file_list


def i_file_search(path=str(Path.home()), recursive=False):
    """Iterative File Search

    Args:
        path (str, optional): Path to file or folder. Defaults to str(Path.home()).

    Returns:
        List: List of files in that directory
    """

    file_list = []
    for file in iglob(pathname=path, recursive=recursive):
        file_list.append(file)
    return file_list


def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data


def read_yaml(filepath):
    if not os.path.exists(filepath):
        filepath = find_file(filepath)
    with io.open(filepath, encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def read_bytes_stream(filepath):
    with open(filepath, mode='rb') as f:
        return io.BytesIO(f.read())

#------------------------File Archiver----------------------#


def unpack_archive(filename, extract_dir):
    extract_dir = Path(extract_dir)
    filename = Path(filename)

    if not extract_dir.exists():
        # create directory if extract directory does not exist
        extract_dir.mkdir(parents=True)

    if filename.exists():

        try:
            shutil.unpack_archive(filename=filename, extract_dir=extract_dir)
        except ValueError as e:

            try:
                # get file extension if shutil cannot parse/find format
                archive_format_list = {
                    "zip": "zip",
                    "gz": "gztar",
                    "bz2": "bztar",
                    "xz": "xztar"
                }
                archive_format = filename.suffix
                if archive_format in archive_format_list.keys():
                    shutil.unpack_archive(
                        filename, extract_dir, format=archive_format_list[archive_format])
                else:
                    log.error("")

            except KeyError as e:
                error_msg = f"{e}: Archive format invalid!"
                log.error(error_msg)
                st.error(error_msg)
        else:
            # pass IOError
            error_msg = f"{IOError}: File does not exist"
            log.error(error_msg)
            st.error(error_msg)

    return f"successfully archived"

# def single_file_archive():

