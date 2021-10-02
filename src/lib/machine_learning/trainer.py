"""
Title: Training Management
Date: 01/10/2021
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
import subprocess
import sys
import shutil
from math import ceil
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional

import pandas as pd
import wget
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state

import tensorflow as tf
import object_detection
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

# >>>> User-defined Modules >>>>
from sklearn.model_selection import train_test_split
from imutils.paths import list_images
from core.utils.file_handler import create_folder_if_not_exist
from core.utils.form_manager import (check_if_exists, check_if_field_empty,
                                     reset_page_attributes)
from core.utils.helper import (NetChange, datetime_formatter, find_net_change,
                               get_directory_name, join_string)
from core.utils.log import logger  # logger
from data_manager.database_manager import (db_fetchall, db_fetchone,
                                           db_no_fetch, init_connection)
from project.project_management import Project
from data_manager.dataset_management import Dataset
from training.model_management import Model, NewModel
from training.training_management import Training
from training.labelmap_management import Framework, Labels
from deployment.deployment_management import Deployment, DeploymentType
from path_desc import (TFOD_DIR, PRE_TRAINED_MODEL_DIR,
                       USER_DEEP_LEARNING_MODEL_UPLOAD_DIR, TFOD_MODELS_TABLE_PATH,
                       CLASSIF_MODELS_NAME_PATH, SEGMENT_MODELS_TABLE_PATH, chdir_root)

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration >>>>

# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])

# *********************PAGINATION**********************


def run_command(command_line_args, st_output: bool = False):
    """
    Running commands or scripts.
    Set `st_output` to True to show the console outputs LIVE on Streamlit.
    """
    # shell=True to work on String instead of list
    logger.info(f"Running command: '{command_line_args}'")
    process = subprocess.run(command_line_args, shell=True,
                             # stdout to capture all output
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             # text to directly decode the output
                             text=True)
    for line in process.stdout:
        if st_output:
            st.markdown(line)
        else:
            print(line)
    process.wait()
    return process.stdout


class Trainer:
    def __init__(self, project: Project, new_training: Training):
        """
        You may choose to inherit this class if you would like to change the __init__ method,
        but make sure to retain all the attributes here for the other methods to work.
        """
        self.project_id: int = project.id
        self.training_id: int = new_training.id
        self.project_path: Path = project.get_project_path()
        self.deployment_type: str = project.deployment_type
        self.class_names: List[str] = project.get_existing_unique_labels(
        ).tolist()
        # with keys: 'train', 'eval', 'test'
        self.partition_ratio: Dict[str, float] = new_training.partition_ratio
        self.dataset_export_path: Path = project.get_export_path()
        self.attached_model_name: str = new_training.attached_model.name
        self.training_model_name: str = new_training.training_model.name
        # NOTE: Might need these later
        # self.attached_model: Model = new_training.attached_model
        # self.training_model: Model = new_training.training_model
        self.training_param: Dict[str, Any] = new_training.training_param_dict
        self.output_paths: Dict[str, Path] = new_training.get_trained_filepaths(
            self.project_path, new_training.name,
            self.training_model_name, self.deployment_type
        )

    @staticmethod
    def train_test_split(image_dir: Path,
                         test_size: float,
                         *,
                         no_validation: bool = False,
                         annotation_dir: Optional[Path] = None,
                         val_size: Optional[float] = 0.0,
                         train_size: Optional[float] = 0.0,
                         random_seed: Optional[int] = 42
                         ):
        """
        Splitting the dataset into train set, test set, and optionally validation set
        if `no_validation` is True.
        NOTE: annotation_dir is not required for image classification only.

        Args:
            image_dir (Path): Directory to the images.
            test_size (float): Size of test set in percentage
            no_validation (bool, optional): If True, only split into train and test sets. Defaults to False.
            annotation_dir (Optional[Path]): Pass in this parameter to also split the annotation paths. Defaults to None.
            val_size (Optional[float]): Size of validation split, only needed if `no_validation` is True. Defaults to 0.0.
            train_size (Optional[float]): This is only used for logging, can be inferred, thus not required. Defaults to 0.0.
            random_seed (Optional[int]): random seed to use for splitting. Defaults to 42.

        Returns:
            Tuples of lists of image paths, and optionally annotation paths, 
            optionally split without validation set too.
        """
        if no_validation:
            assert not val_size, "Set `no_validation` to True if want to split into validation set too."
        else:
            assert val_size, "Must pass in `val_size` if `no_validation` is False."

        # get the image paths and sort them
        image_paths = sorted(list_images(image_dir / "images"))
        logger.info(f"Total images = {len(image_paths)}")

        # directory to annotation folder, only change this path when necessary
        if annotation_dir is not None:
            # get the label paths and sort them to align with image paths
            label_paths = sorted(list(annotation_dir.iterdir()))
        else:
            label_paths = []

        # TODO: maybe can add stratification as an option
        if no_validation:
            train_size = train_size if train_size else round(1 - test_size, 2)
            logger.info("Splitting into train:test dataset"
                        f" with ratio of {train_size:.2f}:{test_size:.2f}")
            if label_paths:
                X_train, X_test, y_train, y_test = train_test_split(
                    image_paths, label_paths,
                    test_size=test_size,
                    random_state=random_seed
                )
                return X_train, X_test, y_train, y_test
            else:
                X_train, X_test = train_test_split(
                    image_paths,
                    test_size=test_size,
                    random_state=random_seed
                )
                return X_train, X_test
        else:
            train_size = train_size if train_size else round(
                1 - test_size - val_size, 2)
            logger.info("Splitting into train:valid:test dataset"
                        " with ratio of "
                        f"{train_size:.2f}:{val_size:.2f}:{test_size:.2f}")
            if label_paths:
                X_train, X_val_test, y_train, y_val_test = train_test_split(
                    image_paths, label_paths,
                    test_size=(val_size + test_size),
                    random_state=random_seed
                )
                X_val, X_test, y_val, y_test = train_test_split(
                    X_val_test, y_val_test,
                    test_size=(test_size / (val_size + test_size)),
                    shuffle=False,
                    random_state=random_seed,
                )
                return X_train, X_val, X_test, y_train, y_val, y_test
            else:
                X_train, X_val_test = train_test_split(
                    image_paths,
                    test_size=(val_size + test_size),
                    random_state=random_seed
                )
                X_val, X_test = train_test_split(
                    X_val_test,
                    test_size=(test_size / (val_size + test_size)),
                    shuffle=False,
                    random_state=random_seed,
                )
                return X_train, X_val, X_test

    @staticmethod
    def copy_images(image_paths: Path, label_paths: Path, dest_dir: Path, data_type: str):
        assert data_type in ("train", "valid", "test")
        image_dest = os.path.join(dest_dir, data_type)

        logger.info(f"Copying files from {data_type} dataset to {image_dest}")
        if image_dest.exists():
            # remove the existing images
            shutil.rmtree(image_dest, ignore_errors=False)

        # create new directories
        os.makedirs(image_dest, exist_ok=True)

        for image_path, label_path in zip(image_paths, label_paths):
            # copy the image file and label file to the new directory
            shutil.copy2(image_path, image_dest)
            shutil.copy2(label_path, image_dest)

    def run_tfod_training(self):
        # TODO: beware of user-uploaded model
        # TODO: consider the option of allowing training of multiple models
        #       within the same training session

        # this name is used for the output model paths, see self.output_paths
        CUSTOM_MODEL_NAME = self.training_model_name
        # this df has columns: Model Name, Speed (ms), COCO mAP, Outputs, model_links
        models_df = pd.read_csv(TFOD_MODELS_TABLE_PATH, usecols=[
                                'Model Name', 'model_links'])
        PRETRAINED_MODEL_URL = models_df.loc(
            models_df['Model Name'] == self.attached_model_name, 'model_links').squeeze()
        # this PRETRAINED_MODEL_DIRNAME is different from self.attached_model_name,
        #  PRETRAINED_MODEL_DIRNAME is the the first folder's name in the downloaded tarfile
        PRETRAINED_MODEL_DIRNAME = PRETRAINED_MODEL_URL.split(
            "/")[-1].split(".tar.gz")[0]
        # this name is based on `generate_labelmap_file` function
        LABEL_MAP_NAME = 'labelmap.pbtxt'

        # TODO: make some of these paths easily obtained through Training methods
        training_path = self.output_paths['training_path']
        paths = {
            'WORKSPACE_PATH': training_path,
            'APIMODEL_PATH': TFOD_DIR,
            'ANNOTATION_PATH': training_path / 'annotations',
            'IMAGE_PATH': training_path / 'images',
            'PRETRAINED_MODEL_PATH': PRE_TRAINED_MODEL_DIR,
            'CHECKPOINT_PATH': self.output_paths['model_path'],
            'OUTPUT_PATH': self.output_paths['model_export_path'],
        }

        files = {
            'PIPELINE_CONFIG': paths['CHECKPOINT_PATH'] / 'pipeline.config',
            # this generate_tfrecord.py script is modified from https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html
            # to convert our PascalVOC XML annotation files into TFRecords which will be used by the TFOD API
            'GENERATE_TF_RECORD': LIB_PATH / "machine_learning" / "module" / "generate_tfrecord_st.py",
            'LABELMAP': paths['ANNOTATION_PATH'] / LABEL_MAP_NAME,
            'MODEL_TARFILE_PATH': self.output_paths['model_tarfile_path'],
        }

        # create all the necessary paths if not exists yet
        for path in paths.values():
            if not os.path.exists(path):
                os.makedirs(path)

        # ************* Generate train & test images in the folder if not exists *************
        if not os.listdir(paths['IMAGE_PATH']):
            with st.spinner('Generating train test splits ...'):
                # for now we only makes use of test set for TFOD to make things simpler
                test_size = self.partition_ratio['eval'] + \
                    self.partition_ratio['test']
                X_train, X_test, y_train, y_test = self.train_test_split(
                    # - BEWARE that the directories might be different if it's user uploaded
                    image_dir=self.dataset_export_path / "images",
                    test_size=test_size,
                    annotation_dir=self.dataset_export_path / "Annotations",
                    no_validation=True
                )

                logger.info(f"Total training images = {len(y_train)}")
                logger.info(f"Total testing images = {len(y_test)}")

                self.copy_images(X_train, y_train,
                                 paths['IMAGE_PATH'], "train")
                self.copy_images(X_test, y_test, paths['IMAGE_PATH'], "test")
                logger.info("Dataset files copied successfully.")

        # ************* get the pretrained model if not exists *************
        with st.spinner('Downloading pretrained model ...'):
            if not (paths['PRETRAINED_MODEL_PATH'] / PRETRAINED_MODEL_DIRNAME).exists():
                wget.download(PRETRAINED_MODEL_URL)
                pretrained_tarfile = PRETRAINED_MODEL_DIRNAME + '.tar.gz'
                # this command will extract the files at the cwd
                run_command(f"tar -zxvf {pretrained_tarfile}")
                shutil.move(PRETRAINED_MODEL_DIRNAME,
                            paths['PRETRAINED_MODEL_PATH'])
                os.remove(pretrained_tarfile)

        # ******************** Create label_map.pbtxt *****************
        with st.spinner('Creating labelmap file ...'):
            CLASS_NAMES = self.class_names
            labelmap_string = Labels.generate_labelmap_string(
                CLASS_NAMES,
                framework=Framework.TensorFlow,
                deployment_type=self.deployment_type)
            Labels.generate_labelmap_file(labelmap_string=labelmap_string,
                                          # labelmap file resides here
                                          dst=paths['ANNOTATION_PATH'],
                                          framework=Framework.TensorFlow,
                                          deployment_type=self.deployment_type)

        # ******************** Generate TFRecords ********************
        with st.spinner('Generating TFRecords'):
            run_command(
                f'python {files["GENERATE_TF_RECORD"]} '
                f'-x {paths["IMAGE_PATH"] / "train"} '
                f'-l {files["LABELMAP"]} '
                f'-o {paths["ANNOTATION_PATH"] / "train.record"}'
            )
            run_command(
                f'python {files["GENERATE_TF_RECORD"]} '
                f'-x {paths["IMAGE_PATH"] / "test"} '
                f'-l {files["LABELMAP"]} '
                f'-o {paths["ANNOTATION_PATH"] / "test.record"}'
            )

        # ********************* pipeline.config *********************
        with st.spinner('Generating pipeline config file ...'):
            original_config_path = paths['PRETRAINED_MODEL_PATH'] / \
                PRETRAINED_MODEL_DIRNAME / 'pipeline.config'
            # copy over the pipeline.config file before modifying it
            shutil.copy2(original_config_path, paths['CHECKPOINT_PATH'])

            # making pipeline.config file editable programatically
            pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
            with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
                proto_str = f.read()
                text_format.Merge(proto_str, pipeline_config)

            # changing model config
            if 'centernet' in PRETRAINED_MODEL_URL:
                pipeline_config.model.center_net.num_classes = len(CLASS_NAMES)
            elif 'ssd' in PRETRAINED_MODEL_URL or 'efficientdet' in PRETRAINED_MODEL_URL:
                pipeline_config.model.ssd.num_classes = len(CLASS_NAMES)
            elif 'faster_rcnn' in PRETRAINED_MODEL_URL:
                pipeline_config.model.faster_rcnn.num_classes = len(
                    CLASS_NAMES)
            else:
                logger.error("Pretrained model config is not found!")

            pipeline_config.train_config.batch_size = self.training_param['batch_size']
            pipeline_config.train_config.fine_tune_checkpoint = (
                paths['PRETRAINED_MODEL_PATH'] / PRETRAINED_MODEL_DIRNAME / 'checkpoint' / 'ckpt-0')
            pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
            pipeline_config.train_input_reader.label_map_path = files['LABELMAP']
            pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
                paths['ANNOTATION_PATH'] / 'train.record']
            pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
            pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
                paths['ANNOTATION_PATH'] / 'test.record']

            config_text = text_format.MessageToString(pipeline_config)
            with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
                f.write(config_text)

        # ************************ TRAINING ************************
        # the path to the training script file `model_main_tf2.py` used to train our model
        st.markdown("Training started ...")
        TRAINING_SCRIPT = paths['APIMODEL_PATH'] / \
            'research' / 'object_detection' / 'model_main_tf2.py'
        # change the training steps as necessary, recommended start with 300 to test whether it's working, then train for at least 2000 steps
        NUM_TRAIN_STEPS = self.training_param['num_train_steps']

        command = (f"python {TRAINING_SCRIPT} "
                   f"--model_dir={paths['CHECKPOINT_PATH']} "
                   f"--pipeline_config_path={files['PIPELINE_CONFIG']} "
                   f"--num_train_steps={NUM_TRAIN_STEPS}")
        run_command(command, st_output=True)

        # TODO: Setup TensorBoard logdir
        # %tensorboard --logdir={paths['CHECKPOINT_PATH']}

        # *********************** EXPORT MODEL ***********************
        with st.spinner("Exporting model ..."):
            FREEZE_SCRIPT = paths['APIMODEL_PATH'] / 'research' / \
                'object_detection' / 'exporter_main_v2.py '
            command = (f"python {FREEZE_SCRIPT} "
                       "--input_type=image_tensor "
                       f"--pipeline_config_path={files['PIPELINE_CONFIG']} "
                       f"--trained_checkpoint_dir={paths['CHECKPOINT_PATH']} "
                       f"--output_directory={paths['OUTPUT_PATH']}")
            run_command(command)

            # Also copy the label map file to the exported directory to use for display
            # label names in inference later
            shutil.copy2(files['LABELMAP'], paths['OUTPUT_PATH'])

            # tar the exported model to be used anywhere
            # NOTE: be careful with the second argument of tar command,
            #  if you pass a chain of directories, the directories
            #  will also be included in the tarfile. That's why we `chdir` first
            os.chdir(paths['OUTPUT_PATH'].parent)
            run_command(
                f"tar -czf {files['MODEL_TARFILE_PATH'].name} {paths['OUTPUT_PATH']}")

            # after created the tarfile in the current working directory,
            #  then only move to the desired filepath
            shutil.move(files['MODEL_TARFILE_PATH'].name,
                        files['MODEL_TARFILE_PATH'])

            logger.info(
                f"CONGRATS YO! Successfully created {files['MODEL_TARFILE_PATH']}")

            # and then change back to root dir
            chdir_root()

        # TODO: INFERENCE
