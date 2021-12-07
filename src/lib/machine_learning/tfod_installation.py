""" 
Title: TensorFlow Object Detection (TFOD) API Installation
Date: 30/9/2021 
Author: Anson Tan Chen Tung
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
import shutil
import stat
import sys
import wget
import subprocess
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

# >>>> User-defined Modules >>>>
from core.utils.log import logger
from path_desc import TFOD_DIR, chdir_root


def run_command(command_line_args):
    logger.info(f"Running command: '{command_line_args}'")
    # shell=True to work on String instead of list
    process = subprocess.Popen(command_line_args, shell=True,
                               # stdout to capture all output
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               # text to directly decode the output
                               text=True)
    for line in process.stdout:
        line = line.strip()
        # print the live stdout output from the script
        if line:
            print(line)
    process.wait()
    return process.stdout


def del_rw(action, name, exc):
    """To delete .git directory in TFOD"""
    os.chmod(name, stat.S_IWRITE)
    os.remove(name)


def install():
    PROTOC_PATH = TFOD_DIR.parent / 'protoc_installation'

    for path in (TFOD_DIR, PROTOC_PATH):
        os.makedirs(path, exist_ok=True)

    if not (TFOD_DIR / 'research' / 'object_detection').exists():
        logger.info(
            "Cloning TFOD API from https://github.com/tensorflow/models...")
        run_command(
            f"git clone https://github.com/tensorflow/models {str(TFOD_DIR)}")

    # Install Tensorflow Object Detection and dependencies such as protobuf and protoc
    # NOTE: Install COCO API ONLY if you want to perform evaluation
    if os.name == 'posix':
        logger.info('Installing COCO API ...')
        cwd = os.getcwd()
        run_command("git clone https://github.com/cocodataset/cocoapi.git")
        os.chdir("cocoapi/PythonAPI")
        run_command("make")
        shutil.copytree("pycocotools", TFOD_DIR / 'research')
        os.chdir(cwd)
        # removed the unused cloned files from COCO API
        shutil.rmtree("cocoapi")
        # 'posix' is for Linux (also to use in Colab Notebook)
        logger.info("Installing protobuf ...")
        run_command(f"apt-get install protobuf-compiler")
        cmd = (f"cd {TFOD_DIR / 'research'} "
               "&& protoc object_detection/protos/*.proto --python_out=. "
               "&& cp object_detection/packages/tf2/setup.py setup.py "
               "&& python setup.py build && python setup.py install")
        run_command(cmd)
        # installing pycocotools here instead of in requirements.txt
        # because Windows NEED to install pycocotools-windows instead
        run_command('pip install pycocotools==2.0.2')
        # The command below does not work properly because the pip install will stuck for very long
        # run_command(
        #     f"cd {TFOD_DIR / 'research'} "
        #     "&& protoc object_detection/protos/*.proto --python_out=. "
        #     "&& cp object_detection/packages/tf2/setup.py . "
        #     "&& python -m pip install . ")
    elif os.name == 'nt':
        # 'nt' is for Windows
        # NOTE: Windows need to install COCO API first, but there is currently an ongoing issue
        # for running COCO evaluation on Windows, refer to here https://github.com/google/automl/issues/487
        # Instructions for COCO API Installation: https://github.com/ansonnn07/object-detection#coco-api-installation
        #  (under the COCO API installation section)
        # NOTE: using this new repo created to fix Windows installation for now
        run_command(
            "pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI")
        if not (PROTOC_PATH / "bin").exists():
            logger.info("Downloading protobuf dependencies ...")
            protoc_version = "3.19.1"  # updated from 3.15.6 -> 3.19.1
            protoc_zipfilename = f"protoc-{protoc_version}-win64.zip"
            url = f"https://github.com/protocolbuffers/protobuf/releases/download/v{protoc_version}/{protoc_zipfilename}"
            wget.download(url)
            # move the protoc zip file into the desired path, PROTOC_PATH
            shutil.move(protoc_zipfilename, PROTOC_PATH)
            # unzip the zip file
            run_command(
                f"cd {PROTOC_PATH} && tar -xf {protoc_zipfilename}")
            os.remove(PROTOC_PATH / protoc_zipfilename)
        # add the path of $PROTOC_PATH/bin into the PATH in environment variable
        # to be able to run `protoc` as a command in terminal
        os.environ['PATH'] += os.pathsep + str((PROTOC_PATH / 'bin').resolve())
        # run the `protoc` command and install all the dependencies for TFOD API
        logger.info("Installing TFOD API ...")
        cmd = (f"cd {TFOD_DIR / 'research'} "
               "&& protoc object_detection/protos/*.proto --python_out=. "
               "&& copy object_detection\\packages\\tf2\\setup.py setup.py "
               "&& python setup.py build "
               "&& python setup.py install")
        run_command(cmd)
        # reason explained above under Linux part
        run_command('pip install pycocotools-windows==2.0.0.2')

    # install slim dependencies
    run_command(f"cd {TFOD_DIR / 'research'}/slim && pip install -e .")

    # remove unnecessary files
    folders_to_del = (TFOD_DIR, (TFOD_DIR / 'research'))
    folders_to_keep = set(('research', 'object_detection', 'pycocotools'))
    for folder in folders_to_del:
        for path in folder.iterdir():
            if path.is_dir():
                # these folders should not be removed
                if path.name in folders_to_keep:
                    continue
                if path.name == '.git':
                    # .git folder needs special permission
                    shutil.rmtree(path, onerror=del_rw)
                else:
                    shutil.rmtree(path)
            else:
                os.remove(path)

    # NOTE: not running verification script for now as the environment might not be updated
    #   with the latest packages yet
    # VERIFICATION_SCRIPT = TFOD_DIR / 'research' / \
    #     'object_detection' / 'builders' / 'model_builder_tf2_test.py'
    # # Verify all the installation above works for TFOD API
    # logger.info("Verifying all works ...")
    # run_command(f"python {VERIFICATION_SCRIPT}")
    # logger.info(
    #     "If the last line above returns 'OK' then TFOD API is installed successfully.")

    chdir_root()


if __name__ == '__main__':
    install()
