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

from functools import partial
from itertools import cycle
import math
import os
import sys
import shutil
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pickle
import glob
import re

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wget
import streamlit as st
from streamlit import session_state
from stqdm import stqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import config_util
from object_detection.builders import model_builder

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils.paths import list_images

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

# >>>> User-defined Modules >>>>
from core.utils.log import logger  # logger
from data_manager.database_manager import init_connection
from project.project_management import Project
from training.training_management import Training, AugmentationConfig
from training.labelmap_management import Framework, Labels
from path_desc import (TFOD_DIR, PRE_TRAINED_MODEL_DIR,
                       USER_DEEP_LEARNING_MODEL_UPLOAD_DIR, TFOD_MODELS_TABLE_PATH,
                       CLASSIF_MODELS_NAME_PATH, SEGMENT_MODELS_TABLE_PATH, chdir_root)
from .utils import (LRTensorBoard, StreamlitOutputCallback, find_tfod_eval_metrics, generate_tfod_xml_csv, get_bbox_label_info, get_transform, run_command, run_command_update_metrics_2,
                    find_tfod_metric, load_image_into_numpy_array, load_labelmap, tfod_ckpt_detect_fn,
                    xml_to_df)
from .visuals import PrettyMetricPrinter, create_class_colors, draw_gt_bbox, draw_tfod_bboxes, pretty_format_param


class Trainer:
    def __init__(self, project: Project, new_training: Training):
        """
        You may choose to inherit this class if you would like to change the __init__ method,
        but make sure to retain all the attributes here for the other methods to work.
        """
        self.project_id: int = project.id
        self.training_id: int = new_training.id
        self.training_model_id: int = new_training.training_model.id
        self.attached_model_name: str = new_training.attached_model.name
        self.training_model_name: str = new_training.training_model.name
        self.project_path: Path = project.get_project_path(project.name)
        self.deployment_type: str = project.deployment_type
        self.class_names: List[str] = project.get_existing_unique_labels(
            project.id)
        # with keys: 'train', 'eval', 'test'
        self.partition_ratio: Dict[str, float] = new_training.partition_ratio
        # if user selected both validation and testing partition ratio, we will have validation set
        self.has_valid_set = True if self.partition_ratio['test'] > 0 else False
        self.dataset_export_path: Path = project.get_export_path()
        # NOTE: Might need these later
        # self.attached_model: Model = new_training.attached_model
        # self.training_model: Model = new_training.training_model
        self.training_param: Dict[str, Any] = new_training.training_param_dict
        self.augmentation_config: AugmentationConfig = new_training.augmentation_config
        self.training_path: Dict[str, Path] = new_training.training_path
        # created this path to save all the test set related paths and encoded_labels
        self.test_set_pkl_path: Path = self.training_path['models'] / \
            'test_images_and_labels.pkl'
        self.test_result_txt_path: Path = self.training_path['models'] / \
            'test_result.txt'
        self.confusion_matrix_path: Path = self.training_path['models'] / \
            'confusion_matrix.png'

    @staticmethod
    def train_test_split(image_paths: List[Path],
                         test_size: float,
                         *,
                         no_validation: bool,
                         labels: Union[List[str], List[Path]],
                         val_size: Optional[float] = 0.0,
                         train_size: Optional[float] = 0.0,
                         stratify: Optional[bool] = False,
                         random_seed: Optional[int] = 42
                         ) -> Tuple[List[str], ...]:
        """
        Splitting the dataset into train set, test set, and optionally validation set
        if `no_validation` is True. Image classification will pass in label names instead
        of label_paths for each image.

        Args:
            image_paths (Path): Directory to the images.
            test_size (float): Size of test set in percentage
            no_validation (bool): If True, only split into train and test sets, without validation set.
            labels (Union[str, Path]): Pass in this parameter to split the labels or label paths.
            val_size (Optional[float]): Size of validation split, only needed if `no_validation` is False. Defaults to 0.0.
            train_size (Optional[float]): This is only used for logging, can be inferred, thus not required. Defaults to 0.0.
            stratify (Optional[bool]): stratification should only be used for image classification. Defaults to False
            random_seed (Optional[int]): random seed to use for splitting. Defaults to 42.

        Returns:
            Tuples of lists of image paths (str), and optionally annotation paths,
            optionally split without validation set too.
        """
        if no_validation:
            assert not val_size, "Set `no_validation` to True if want to split into validation set too."
        else:
            assert val_size, "Must pass in `val_size` if `no_validation` is False."

        total_images = len(image_paths)
        assert total_images == len(labels)

        if stratify:
            depl_type = session_state.new_training.deployment_type
            assert depl_type == 'Image Classification', (
                'Only use stratification for image classification labels. '
                f'Current deployment type: {depl_type}'
            )
            stratify = labels
        else:
            stratify = None

        # get the image paths and sort them
        logger.info(f"Total images = {total_images}")

        # TODO: maybe can add stratification as an option (only works for img classification)
        if no_validation:
            train_size = train_size if train_size else round(1 - test_size, 2)
            logger.info("Splitting into train:test dataset"
                        f" with ratio of {train_size:.2f}:{test_size:.2f}")
            X_train, X_test, y_train, y_test = train_test_split(
                image_paths, labels,
                test_size=test_size,
                stratify=stratify,
                random_state=random_seed
            )
            return X_train, X_test, y_train, y_test
        else:
            train_size = train_size if train_size else round(
                1 - test_size - val_size, 2)
            logger.info("Splitting into train:valid:test dataset"
                        " with ratio of "
                        f"{train_size:.2f}:{val_size:.2f}:{test_size:.2f}")
            X_train, X_val_test, y_train, y_val_test = train_test_split(
                image_paths, labels,
                test_size=(val_size + test_size),
                stratify=stratify,
                random_state=random_seed
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_val_test, y_val_test,
                test_size=(test_size / (val_size + test_size)),
                shuffle=False,
                stratify=stratify,
                random_state=random_seed,
            )
            return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def copy_images(image_paths: Path,
                    dest_dir: Path,
                    label_paths: Optional[Path] = None):
        if dest_dir.exists():
            # remove the existing images
            shutil.rmtree(dest_dir, ignore_errors=False)

        # create new directories
        os.makedirs(dest_dir)

        if label_paths:
            for image_path, label_path in stqdm(zip(image_paths, label_paths), total=len(image_paths)):
                # copy the image file and label file to the new directory
                shutil.copy2(image_path, dest_dir)
                shutil.copy2(label_path, dest_dir)
        else:
            for image_path in image_paths:
                shutil.copy2(image_path, dest_dir)

    def train(self, is_resume: bool = False, stdout_output: bool = False):
        logger.debug("Clearing all existing Streamlit cache")
        # clearing all cache in case there is something weird happen with the
        # st.experimental_memo methods
        st.legacy_caching.clear_cache()

        with st.spinner("Exporting tasks for training ..."):
            session_state.project.export_tasks(
                for_training_id=session_state.new_training.id)

        if self.training_path['export'].exists:
            # remove the exported model first before training
            shutil.rmtree(self.training_path['export'])

        logger.info(f"Start training for Training {self.training_id}")
        if self.deployment_type == 'Object Detection with Bounding Boxes':
            if not is_resume:
                self.reset_tfod_progress()
            else:
                progress = session_state.new_training.progress
                progress['Checkpoint'] += 1
                session_state.new_training.update_progress(progress)
            self.run_tfod_training(stdout_output)
        elif self.deployment_type == "Image Classification":
            if not is_resume:
                self.reset_keras_progress()
            self.run_keras_training(is_resume=is_resume, classification=True)
        elif self.deployment_type == "Semantic Segmentation with Polygons":
            # TODO: train image segmentation model
            # self.run_keras_training(is_resume=is_resume, segmentation=True)
            pass

    def evaluate(self):
        logger.info(f"Start evaluation for Training {self.training_id}")
        if self.deployment_type == 'Object Detection with Bounding Boxes':
            self.run_tfod_evaluation()
        elif self.deployment_type == "Image Classification":
            self.run_classification_eval()
        elif self.deployment_type == "Semantic Segmentation with Polygons":
            pass

    def export_model(self):
        # msg_container is generated from st.container to show message under
        # the Export button
        logger.info(f"Exporting model for Training ID {self.training_id}, "
                    f"Model ID {self.training_model_id}")
        if self.deployment_type == 'Object Detection with Bounding Boxes':
            self.export_tfod_model(stdout_output=False)
        else:
            self.export_keras_model()

    def reset_tfod_progress(self):
        # reset the training progress
        training_progress = {'Step': 0, 'Checkpoint': 0}
        session_state.new_training.reset_training_progress(
            training_progress)

    def reset_keras_progress(self):
        # reset the training progress
        training_progress = {'Epoch': 0}
        session_state.new_training.reset_training_progress(
            training_progress)

    def tfod_update_progress_metrics(self, cmd_output: str) -> Dict[str, float]:
        # ! DEPRECATED, update progress in real time during training using `run_command_update_metrics_2`
        """
        This function updates only the latest metrics from the **entire** stdout,
        instead of only from a single line. So this does not work for continuously
        updating the database during training.
        Refer 'run_command_update_metrics' function for continuous updates.
        """
        step = find_tfod_metric("Step", cmd_output)
        local_loss = find_tfod_metric(
            "Loss/localization_loss", cmd_output)
        reg_loss = find_tfod_metric(
            "Loss/regularization_loss", cmd_output)
        total_loss = find_tfod_metric(
            "Loss/total_loss", cmd_output)
        learning_rate = find_tfod_metric(
            "learning_rate", cmd_output)
        metrics = {
            "localization_loss": float(local_loss),
            "regularization_loss": float(reg_loss),
            "total_loss": float(total_loss),
            "learning_rate": float(learning_rate)
        }
        session_state.new_training.update_progress(step=int(step))
        session_state.new_training.update_metrics(metrics)
        return metrics

    def get_tfod_last_ckpt_path(self) -> Path:
        """Find and return the latest TFOD checkpoint path.
        Return None if no ckpt-*.index file found"""
        ckpt_dir = self.training_path['models']
        ckpt_filepaths = glob.glob(str(ckpt_dir / 'ckpt-*.index'))
        if not ckpt_filepaths:
            logger.warning("""There is no checkpoint file found,
            the TFOD model is not trained yet.""")
            return None

        def get_ckpt_cnt(path):
            ckpt = path.split("ckpt-")[-1].split(".")[0]
            return int(ckpt)

        latest_ckpt = sorted(ckpt_filepaths, key=get_ckpt_cnt, reverse=True)[0]
        return Path(latest_ckpt)

    def check_model_exists(self) -> Dict[str, bool]:
        """Check whether model weights (aka checkpoint) and exported model tarfile exists.
        This is to check whether our model needs to be exported first before letting the
        user to download."""
        ckpt_found = model_tarfile_found = False
        if self.deployment_type == 'Image Classification':
            ckpt_path = self.training_path['keras_model_file']
        elif self.deployment_type == 'Object Detection with Bounding Boxes':
            ckpt_path = self.get_tfod_last_ckpt_path()

        if ckpt_path and ckpt_path.exists():
            ckpt_found = True
        if self.training_path['model_tarfile'].exists():
            model_tarfile_found = True

        return {'ckpt': ckpt_found,
                'model_tarfile': model_tarfile_found}

    def run_tfod_training(self, stdout_output: bool = False):
        """
        Run training for TensorFlow Object Detection (TFOD) API.
        Can be used for Mask R-CNN model for image segmentation if wanted.
        Set `stdout_output` to True to print out the long console outputs
        generated from the script.
        """
        # training_param only consists of 'batch_size' and 'num_train_steps'
        # TODO: beware of user-uploaded model

        # this name is used for the output model paths, see self.training_path
        CUSTOM_MODEL_NAME = self.training_model_name
        # this df has columns: Model Name, Speed (ms), COCO mAP, Outputs, model_links
        models_df = pd.read_csv(TFOD_MODELS_TABLE_PATH, usecols=[
                                'Model Name', 'model_links'])
        PRETRAINED_MODEL_URL = models_df.loc[
            models_df['Model Name'] == self.attached_model_name,
            'model_links'].squeeze()
        # this PRETRAINED_MODEL_DIRNAME is different from self.attached_model_name,
        #  PRETRAINED_MODEL_DIRNAME is the the first folder's name in the downloaded tarfile
        PRETRAINED_MODEL_DIRNAME = PRETRAINED_MODEL_URL.split(
            "/")[-1].split(".tar.gz")[0]

        # can check out initialise_training_folder function too
        paths = {
            'WORKSPACE_PATH': self.training_path['ROOT'],
            'APIMODEL_PATH': TFOD_DIR,
            'ANNOTATION_PATH': self.training_path['annotations'],
            'IMAGE_PATH': self.training_path['images'],
            'PRETRAINED_MODEL_PATH': PRE_TRAINED_MODEL_DIR,
            'MODELS': self.training_path['models'],
            'EXPORT': self.training_path['export'],
        }

        files = {
            'PIPELINE_CONFIG': paths['MODELS'] / 'pipeline.config',
            # this generate_tfrecord.py script is modified from https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html
            # to convert our PascalVOC XML annotation files into TFRecords which will be used by the TFOD API
            'GENERATE_TF_RECORD': LIB_PATH / "machine_learning" / "module" / "generate_tfrecord_st.py",
            'LABELMAP': self.training_path['labelmap_file'],
            'MODEL_TARFILE': self.training_path['model_tarfile'],
        }

        # create all the necessary paths if not exists yet
        for path in paths.values():
            if not os.path.exists(path):
                os.makedirs(path)

        # ************* Generate train & test images in the folder *************
        with st.spinner('Generating train test splits ...'):
            image_paths = sorted(list_images(
                self.dataset_export_path / "images"))
            labels = sorted(glob.glob(
                str(self.dataset_export_path / "Annotations" / "*.xml")))
            # for now we only makes use of test set for TFOD to make things simpler
            test_size = self.partition_ratio['eval'] + \
                self.partition_ratio['test']
            X_train, X_test, y_train, y_test = self.train_test_split(
                # - BEWARE that the directories might be different if it's user uploaded
                image_paths=image_paths,
                test_size=test_size,
                labels=labels,
                no_validation=True
            )

        col, _ = st.columns([1, 1])
        with col:
            if self.augmentation_config.train_size is not None:
                # only train set is affected by augmentation, because we are generating
                # our own augmented images before training for TFOD
                train_size = self.augmentation_config.train_size
            else:
                train_size = len(y_train)
            st.code(f"Total training images = {train_size}  \n"
                    f"Total testing images = {len(y_test)}")

        # initialize to check whether these paths exist for generating TF Records
        train_xml_csv_path = None
        with st.spinner('Copying images to folder, this may take awhile ...'):
            if self.augmentation_config.exists():
                with st.spinner("Generating augmented training images ..."):
                    # these csv files are temporarily generated to use for generating TF Records, should be removed later
                    train_xml_csv_path = paths['ANNOTATION_PATH'] / 'train.csv'
                    generate_tfod_xml_csv(
                        image_paths=X_train,
                        xml_dir=self.dataset_export_path / "Annotations",
                        output_img_dir=paths['IMAGE_PATH'] / 'train',
                        csv_path=train_xml_csv_path,
                        train_size=self.augmentation_config.train_size
                    )
            else:
                # if not augmenting data, directly copy the train images to the folder
                self.copy_images(X_train,
                                 dest_dir=paths['IMAGE_PATH'] / "train",
                                 label_paths=y_train)
            # test set images should not be augmented
            self.copy_images(X_test,
                             dest_dir=paths['IMAGE_PATH'] / "test",
                             label_paths=y_test)
            logger.info("Dataset files copied successfully.")

        # ************* get the pretrained model if not exists *************
        if not (paths['PRETRAINED_MODEL_PATH'] / PRETRAINED_MODEL_DIRNAME).exists():
            with st.spinner('Downloading pretrained model ...'):
                wget.download(PRETRAINED_MODEL_URL)
                pretrained_tarfile = PRETRAINED_MODEL_DIRNAME + '.tar.gz'
                # this command will extract the files at the cwd
                run_command(f"tar -zxvf {pretrained_tarfile}")
                shutil.move(PRETRAINED_MODEL_DIRNAME,
                            paths['PRETRAINED_MODEL_PATH'])
                os.remove(pretrained_tarfile)
                logger.info(
                    f'{PRETRAINED_MODEL_DIRNAME} downloaded successfully')

        # ******************** Create label_map.pbtxt *****************
        with st.spinner('Creating labelmap file ...'):
            CLASS_NAMES = self.class_names
            labelmap_string = Labels.generate_labelmap_string(
                CLASS_NAMES,
                framework=Framework.TensorFlow,
                deployment_type=self.deployment_type)
            Labels.generate_labelmap_file(labelmap_string=labelmap_string,
                                          dst=files["LABELMAP"].parent,
                                          framework=Framework.TensorFlow,
                                          deployment_type=self.deployment_type)

        # ******************** Generate TFRecords ********************
        with st.spinner('Generating TFRecords ...'):
            if train_xml_csv_path is not None:
                # using the CSV file generated during the augmentation process above
                # Must provide both image_dir and csv_path to skip the `xml_to_df` conversion step
                run_command(
                    f'python {files["GENERATE_TF_RECORD"]} '
                    f'-i {paths["IMAGE_PATH"] / "train"} '
                    f'-l {files["LABELMAP"]} '
                    f'-d {train_xml_csv_path} '
                    f'-o {paths["ANNOTATION_PATH"] / "train.record"} '
                )
            else:
                run_command(
                    f'python {files["GENERATE_TF_RECORD"]} '
                    f'-x {paths["IMAGE_PATH"] / "train"} '
                    f'-l {files["LABELMAP"]} '
                    f'-o {paths["ANNOTATION_PATH"] / "train.record"}'
                )
            # test set images are not augmented
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
            shutil.copy2(original_config_path, paths['MODELS'])

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
            elif 'faster_rcnn' in PRETRAINED_MODEL_URL or 'mask_rcnn' in PRETRAINED_MODEL_URL:
                pipeline_config.model.faster_rcnn.num_classes = len(
                    CLASS_NAMES)
            else:
                logger.error("Pretrained model config is not found!")
                st.error(f"Error with pretrained model: {self.attached_model_name}. "
                         "Please try selecting another model and train again.")
                time.sleep(5)
                st.experimental_rerun()

            pipeline_config.train_config.batch_size = self.training_param['batch_size']
            pipeline_config.train_config.fine_tune_checkpoint = str(
                paths['PRETRAINED_MODEL_PATH'] / PRETRAINED_MODEL_DIRNAME
                / 'checkpoint' / 'ckpt-0')
            pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
            pipeline_config.train_input_reader.label_map_path = str(
                files['LABELMAP'])
            pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
                str(paths['ANNOTATION_PATH'] / 'train.record')]
            pipeline_config.eval_input_reader[0].label_map_path = str(
                files['LABELMAP'])
            pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
                str(paths['ANNOTATION_PATH'] / 'test.record')]

            config_text = text_format.MessageToString(pipeline_config)
            with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
                f.write(config_text)

        # ************************ TRAINING ************************
        # the path to the training script file `model_main_tf2.py` used to train our model

        TRAINING_SCRIPT = paths['APIMODEL_PATH'] / \
            'research' / 'object_detection' / 'model_main_tf2.py'
        # change the training steps as necessary, recommended start with 300 to test whether it's working, then train for at least 2000 steps
        NUM_TRAIN_STEPS = self.training_param['num_train_steps']

        start = time.perf_counter()
        command = (f"python {TRAINING_SCRIPT} "
                   f"--model_dir={paths['MODELS']} "
                   f"--pipeline_config_path={files['PIPELINE_CONFIG']} "
                   f"--num_train_steps={NUM_TRAIN_STEPS}")
        with st.spinner("**Training started ... This might take awhile ... "
                        "Do not refresh the page **"):
            run_command_update_metrics_2(
                command, stdout_output=stdout_output,
                step_name='Step')
            # cmd_output = run_command(command, st_output=True,
            #                          filter_by=['Step', 'Loss/', 'learning_rate'])
            # self.tfod_update_progress_metrics(cmd_output)
        time_elapsed = time.perf_counter() - start
        m, s = divmod(time_elapsed, 60)
        m, s = int(m), int(s)
        logger.info(f'Finished training! Took {m}m {s}s')
        st.success(f'Model has finished training! Took **{m}m {s}s**')

        # ************************ EVALUATION ************************
        with st.spinner("Running object detection evaluation ..."):
            command = (f"python {TRAINING_SCRIPT} "
                       f"--model_dir={paths['MODELS']} "
                       f"--pipeline_config_path={files['PIPELINE_CONFIG']} "
                       f"--checkpoint_dir={paths['MODELS']}")
            filtered_outputs = run_command(
                command, stdout_output=True, st_output=False,
                filter_by=['DetectionBoxes_', 'Loss/'], is_cocoeval=True)
        eval_result_text = find_tfod_eval_metrics(filtered_outputs)

        st.subheader("Object detection evaluation results on test set:")
        st.info(eval_result_text)

        # store the results to directly show on the training page in the future
        with open(self.test_result_txt_path, 'w') as f:
            f.write(eval_result_text)
        logger.debug("Saved evaluation script results at: "
                     f"{self.test_result_txt_path}")

        # Delete unwanted files excluding those needed for evaluation and exporting
        paths_to_del = (paths['ANNOTATION_PATH'],
                        paths['IMAGE_PATH'] / 'train')
        for p in paths_to_del:
            logger.debug("Removing unwanted directories used only "
                         f"for TFOD training: {p}")
            shutil.rmtree(p)

    def export_tfod_model(self, stdout_output=False):
        paths = self.training_path
        if paths['export'].exists():
            # remove any existing export directory first
            shutil.rmtree(paths['export'])

        with st.spinner("Exporting model ... This may take awhile ..."):
            pipeline_conf_path = paths['models'] / 'pipeline.config'
            FREEZE_SCRIPT = TFOD_DIR / 'research' / \
                'object_detection' / 'exporter_main_v2.py '
            command = (f"python {FREEZE_SCRIPT} "
                       "--input_type=image_tensor "
                       f"--pipeline_config_path={pipeline_conf_path} "
                       f"--trained_checkpoint_dir={paths['models']} "
                       f"--output_directory={paths['export']}")
            run_command(command, stdout_output)
        st.success('Model is successfully exported!')

        # copy the labelmap file into the export directory first
        # to store it for exporting
        shutil.copy2(paths['labelmap_file'], paths['export'])

        # tar the exported model to be used anywhere
        # NOTE: be careful with the second argument of tar -czf command,
        #  if you pass a chain of directories, the directories
        #  will also be included in the tarfile.
        # That's why we `chdir` first and pass only the file/folder names
        with st.spinner("Creating tarfile to download ..."):
            os.chdir(paths['export'].parent)
            run_command(
                f"tar -czf {paths['model_tarfile'].name} "
                f"{paths['export'].name}")
            # and then change back to root dir
            chdir_root()

        logger.info(
            f"CONGRATS YO! Successfully created {paths['model_tarfile']}")

    @st.cache(show_spinner=False)
    def load_tfod_checkpoint(self):
        """
        Loading from checkpoint instead of the exported savedmodel.
        """
        pipeline_conf_path = self.training_path['models'] / 'pipeline.config'
        ckpt_path = self.get_tfod_last_ckpt_path()

        logger.info(f'Loading TFOD checkpoint model ID {self.training_model_id} '
                    f'for Training ID {self.training_id} '
                    f'from {ckpt_path} ...')
        start_time = time.perf_counter()

        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(
            pipeline_conf_path)
        model_config = configs['model']
        detection_model = model_builder.build(
            model_config=model_config, is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        # need to remove the .index extension at the end
        ckpt.restore(str(ckpt_path).strip('.index')).expect_partial()

        @tf.function
        def detect_fn(image):
            """Detect objects in image."""

            image, shapes = detection_model.preprocess(image)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict, shapes)

            return detections

        end_time = time.perf_counter()
        logger.info(f'Done! Took {end_time - start_time:.2f} seconds')
        return detect_fn

    @st.cache(show_spinner=False)
    def load_tfod_model(self):
        """
        NOTE: Caching is used on this method to avoid long loading times.
        Due to this, this method should not be used outside of
        training/deployment page. Hence, it's not defined as a staticmethod.
        Maybe can improve this by using st.experimental_memo or other methods. Not sure.
        """
        # Loading the exported model from the saved_model directory
        saved_model_path = self.training_path['export'] / "saved_model"

        logger.info(f'Loading model ID {self.training_model_id} '
                    f'for Training ID {self.training_id} '
                    f'from {saved_model_path} ...')
        start_time = time.perf_counter()
        # LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
        detect_fn = tf.saved_model.load(str(saved_model_path))
        end_time = time.perf_counter()
        logger.info(f'Done! Took {end_time - start_time:.2f} seconds')
        return detect_fn

    def tfod_detect(self,
                    detect_fn: Any,
                    image_np: np.ndarray,
                    tensor_dtype=tf.uint8) -> Dict[str, Any]:
        """
        `detect_fn` is obtained using `load_tfod_model` or `load_tfod_checkpoint` method. 
        `tensor_dtype` should be `tf.uint8` for exported model; 
        and `tf.float32` for checkpoint model to work. 
        """
        # Running the infernce on the image specified in the  image path
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        # input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        # input_tensor = input_tensor[tf.newaxis, ...]
        # input_tensor = tf.expand_dims(input_tensor, 0)

        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=tensor_dtype)

        # running detection using the loaded model: detect_fn
        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(
            np.int64)
        return detections

    def run_tfod_evaluation(self):
        """Due to some ongoing issues with using COCO API for evaluation on Windows
        (refer https://github.com/google/automl/issues/487),
        I decided not to use the evaluation script for TFOD. Thus, the manual
        evaluation here only shows some output images with bounding boxes"""
        # get the required paths
        paths = self.training_path
        exist_dict = self.check_model_exists()
        with st.spinner("Loading model ... This might take awhile ..."):
            # only use the full exported model after the user has exported it
            # NOTE: take note of the tensor_dtype to convert to work in both ways
            if exist_dict['model_tarfile']:
                detect_fn = self.load_tfod_model()
                tensor_dtype = tf.uint8
            else:
                detect_fn = self.load_tfod_checkpoint()
                tensor_dtype = tf.float32
        category_index = load_labelmap(paths['labelmap_file'])

        # show the stored evaluation result during training
        if self.test_result_txt_path.exists():
            st.subheader("Object Detection Evaluation results")
            with open(self.test_result_txt_path) as f:
                eval_result_text = f.read()
            st.info(eval_result_text)

        # **************** SHOW SOME IMAGES FOR EVALUATION ****************
        st.subheader("Prediction Results on Test Set:")
        options_col, _ = st.columns([1, 1])
        prev_btn_col_1, next_btn_col_1, _ = st.columns([1, 1, 3])
        true_img_col, pred_img_col = st.columns([1, 1])
        prev_btn_col_2, next_btn_col_2, _ = st.columns([1, 1, 3])

        if 'start_idx' not in session_state:
            # to keep track of the image index to show
            session_state['start_idx'] = 0

        @ st.experimental_memo
        def get_test_images_annotations():
            with st.spinner("Getting images and annotations ..."):
                test_data_dir = paths['images'] / 'test'
                logger.debug(f"Test set image directory: {test_data_dir}")
                test_img_paths = sorted(list_images(test_data_dir))
                # get the ground truth bounding box data from XML files
                gt_xml_df = xml_to_df(str(test_data_dir))
            return test_img_paths, gt_xml_df

        # take the test set images
        image_paths, gt_xml_df = get_test_images_annotations()

        with options_col:
            def reset_start_idx():
                session_state['start_idx'] = 0

            st.number_input(
                "Number of samples to show:",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                format='%d',
                key='n_samples',
                help="Number of samples to run detection and display results.",
                on_change=reset_start_idx
            )
            st.number_input(
                "Minimum confidence threshold:",
                min_value=0.1,
                max_value=0.99,
                value=0.6,
                step=0.01,
                format='%.2f',
                key='conf_threshold',
                help=("If a prediction's confidence score exceeds this threshold, "
                      "then it will be displayed, otherwise discarded."),
            )

            st.info(f"**Total test set images**: {len(image_paths)}")
            if exist_dict['ckpt']:
                st.warning("""You are using the model loaded from a
                checkpoint, the inference time will be slightly slower
                than an exported model.  You may use this to check the
                performance of the model on real images before deciding
                to export the model. Or you may choose to export the
                model by clicking the "Export Model" button above first
                before checking the evaluation results. Note that the
                inference time for the first image will take significantly
                longer than others. In production, we will also export the
                model before using it for inference.""")

        n_samples = session_state['n_samples']
        start_idx = session_state['start_idx']
        current_image_paths = image_paths[start_idx: start_idx + n_samples]

        def previous_samples():
            if session_state['start_idx'] > 0:
                session_state['start_idx'] -= n_samples

        def next_samples():
            max_start = len(image_paths) - n_samples
            if session_state['start_idx'] < max_start:
                session_state['start_idx'] += n_samples

        with prev_btn_col_1:
            st.button('⏮️ Previous samples', key='btn_prev_images_1',
                      on_click=previous_samples)
        with next_btn_col_1:
            st.button('Next samples ⏭️', key='btn_next_images_1',
                      on_click=next_samples)

        logger.info(f"Detecting from the test set images: {start_idx}"
                    f" to {start_idx + n_samples} ...")
        with st.spinner("Running detections ..."):
            # create the colors for each class to draw the bboxes nicely
            class_colors = create_class_colors(self.class_names)

            for i, p in enumerate(current_image_paths):
                # current_img_idx = start_idx + i + 1
                logger.debug(f"Detecting on image at: {p}")
                img = load_image_into_numpy_array(str(p))

                filename = os.path.basename(p)
                class_names, bboxes = get_bbox_label_info(gt_xml_df, filename)

                start_t = time.perf_counter()
                detections = self.tfod_detect(detect_fn, img,
                                              tensor_dtype=tensor_dtype)
                time_elapsed = time.perf_counter() - start_t
                logger.info(f"Done inference on {filename}. "
                            f"[{time_elapsed:.2f} secs]")

                img_with_detections = draw_tfod_bboxes(
                    detections, img, category_index,
                    session_state.conf_threshold)

                with true_img_col:
                    img = draw_gt_bbox(img, bboxes,
                                       class_names=class_names,
                                       class_colors=class_colors)
                    st.image(img, caption=f'Ground Truth: {filename}')
                with pred_img_col:
                    st.image(img_with_detections,
                             caption=f'Prediction: {filename}')

        with prev_btn_col_2:
            st.button('⏮️ Previous samples', key='btn_prev_images_2',
                      on_click=previous_samples)
        with next_btn_col_2:
            st.button('Next samples ⏭️', key='btn_next_images_2',
                      on_click=next_samples)

    # ********************* METHODS FOR KERAS TRAINING *********************

    def create_tf_dataset(self, X_train: List[str], y_train: List[str],
                          X_test: List[str], y_test: List[str],
                          X_val: List[str] = None,
                          y_val: List[str] = None):
        # TODO: create tf_dataset for segmentation

        def read_image(imagePath, label, image_size):
            raw = tf.io.read_file(imagePath)
            image = tf.io.decode_image(
                raw, channels=3, expand_animations=False)
            image = tf.image.resize(image, (image_size, image_size))
            image = tf.cast(image / 255.0, tf.float32)
            label = tf.cast(label, dtype=tf.int32)
            return image, label

        image_size = self.training_param['image_size']
        resize_read_image = partial(read_image, image_size=image_size)

        # get the Albumentations transform
        transform = get_transform()

        if self.deployment_type == 'Image Classification':
            # https://albumentations.ai/docs/examples/tensorflow-example/
            def aug_fn(image):
                aug_data = transform(image=image)
                aug_img = aug_data["image"]
                return aug_img

            def augment(image, label):
                aug_img = tf.numpy_function(
                    func=aug_fn, inp=[image], Tout=tf.float32)
                return aug_img, label
        else:
            def aug_fn(image, mask):
                aug_data = transform(image=image, mask=mask)
                aug_img = aug_data["image"]
                aug_mask = aug_data["mask"]
                return aug_img, aug_mask

            def augment(image, mask):
                aug_img, aug_mask = tf.numpy_function(
                    func=aug_fn, inp=[image, mask], Tout=tf.float32)
                return aug_img, aug_mask

        def set_shapes(img, label, img_shape=(image_size, image_size, 3)):
            img.set_shape(img_shape)
            label.set_shape([])
            return img, label

        # only train set is augmented and shuffled
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = (
            train_ds.map(resize_read_image,
                         num_parallel_calls=AUTOTUNE)
            .map(augment, num_parallel_calls=AUTOTUNE)
            .map(set_shapes, num_parallel_calls=AUTOTUNE)
            .shuffle(len(X_train))
            .cache()
            .batch(self.training_param['batch_size'])
            .prefetch(AUTOTUNE)
        )

        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = (
            test_ds.map(resize_read_image,
                        num_parallel_calls=AUTOTUNE)
            .map(set_shapes, num_parallel_calls=AUTOTUNE)
            .cache()
            .batch(self.training_param['batch_size'])
            .prefetch(AUTOTUNE)
        )

        if X_val:
            val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            val_ds = (
                val_ds.map(resize_read_image,
                           num_parallel_calls=AUTOTUNE)
                .map(set_shapes, num_parallel_calls=AUTOTUNE)
                .cache()
                .batch(self.training_param['batch_size'])
                .prefetch(AUTOTUNE)
            )
            return train_ds, val_ds, test_ds

        return train_ds, test_ds

    def build_classification_model(self):
        # e.g. ResNet50
        pretrained_model_func = getattr(tf.keras.applications,
                                        self.attached_model_name)

        image_size = self.training_param['image_size']
        baseModel = pretrained_model_func(
            weights="imagenet", include_top=False,
            input_tensor=keras.Input(shape=(image_size, image_size, 3))
        )

        # construct the head of the model that will be placed on top of the
        # the base model
        headModel = baseModel.output
        # then add extra layers to suit our choice
        headModel = layers.AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = layers.Flatten(name="flatten")(headModel)
        headModel = layers.Dense(256, activation="relu")(headModel)
        headModel = layers.Dropout(0.5)(headModel)
        # the last layer is the most important to ensure the model outputs
        #  the result that we want
        headModel = layers.Dense(
            len(self.class_names), activation="softmax")(headModel)

        # place the head FC model on top of the base model (this will become
        # the actual model we will train)
        model = keras.Model(inputs=baseModel.input, outputs=headModel)

        # freeze the pretrained model
        baseModel.trainable = False

        optimizer_func = getattr(tf.keras.optimizers,
                                 self.training_param['optimizer'])
        lr = self.training_param['learning_rate']

        # using a simple decay for now, can use cosine annealing if wanted
        opt = optimizer_func(learning_rate=lr,
                             decay=lr / self.training_param['num_epochs'])
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=opt, metrics=["accuracy"])
        return model

    def create_callbacks(self, train_size: int,
                         progress_placeholder: Dict[str, Any],
                         num_epochs: int) -> List[Callback]:
        # this callback saves the checkpoint with the best `val_loss`
        ckpt_cb = ModelCheckpoint(
            filepath=self.training_path['model_weights_file'],
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )

        # maybe early_stopping is not necessary
        # early_stopping_cb = EarlyStopping(patience=10, monitor='val_loss',
        #                                   mode='min', restore_best_weights=True)

        tensorboard_cb = LRTensorBoard(
            log_dir=self.training_path['tensorboard_logdir'])

        pretty_metric_printer = PrettyMetricPrinter()
        steps_per_epoch = math.ceil(
            train_size / self.training_param['batch_size'])
        st_output_cb = StreamlitOutputCallback(
            pretty_metric_printer,
            num_epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            progress_placeholder=progress_placeholder,
            refresh_rate=20
        )

        return [ckpt_cb, tensorboard_cb, st_output_cb]

    def load_model_weights(self, build_model: bool = False, model: keras.Model = None):
        """Load the model weights for evaluation or inference. 
        If `build_model` is `True`, the model will be rebuilt and load back the weights.
        Or just send in a built model to directly use it to load the weights"""
        if build_model:
            if self.deployment_type == 'Image Classification':
                model = self.build_classification_model()
            # TODO load weights for segmentation model
        if model is not None:
            model.load_weights(self.training_path['model_weights_file'])
            return model
        else:
            st.error(
                "Model is not loaded. Some error has occurred. Please try again.")
            logger.error(f"""Model {self.training_model_id} is not loaded, please pass
            in a `model` or set `build_model` to True to rebuild the model layers.""")
            st.stop()

    @st.cache(show_spinner=False)
    def load_keras_model(self):
        """load the exported keras model instead of only weights
        Using `_self` instead of `self` to avoid hashing it for memo"""
        model = keras.models.load_model(
            self.training_path['keras_model_file'])
        return model

    def run_keras_training(self,
                           is_resume: bool = False,
                           classification: bool = False,
                           segmentation: bool = False):
        assert any((classification, segmentation))

        if classification:
            # getting the CSV file exported from the dataset
            # the CSV file generated has these columns:
            # image, id, label, annotator, annotation_id, created_at, updated_at, lead_time.
            # The first col `image` contains the absolute paths to the images
            dataset_df = pd.read_csv(self.dataset_export_path / 'result.csv')
            image_paths = dataset_df['image'].values.astype(str)
            print(f"\n{os.path.exists(image_paths[0]) = }")
            label_series = dataset_df['label'].astype('category')
            encoded_labels = label_series.cat.codes.values
            # use this to store the classnames with their indices as keys
            encoded_label_dict = dict(enumerate(label_series.cat.categories))
        elif segmentation:
            # TODO: segmentation training
            pass

        # ************* Generate train & test images in the folder *************
        if self.has_valid_set:
            with st.spinner('Generating train test splits ...'):
                X_train, X_val, X_test, y_train, y_val, y_test = self.train_test_split(
                    # - BEWARE that the directories might be different if it's user uploaded
                    image_paths=image_paths,
                    test_size=self.partition_ratio['test'],
                    val_size=self.partition_ratio['eval'],
                    labels=encoded_labels,
                    no_validation=False
                )

            col, _ = st.columns([1, 1])
            with col:
                st.code(f"Total training images = {len(X_train)}  \n"
                        f"Total validation images = {len(X_val)}"
                        f"Total testing images = {len(X_test)}")
        else:
            # the user did not select a test size, i.e. only using validation set for testing,
            # thus for our train_test_split implementation, we assume we only
            # want to use the test_size without val_size (sorry this might sound confusing)
            with st.spinner('Generating train test splits ...'):
                X_train, X_test, y_train, y_test = self.train_test_split(
                    # - BEWARE that the directories might be different if it's user uploaded
                    image_paths=image_paths,
                    test_size=self.partition_ratio['eval'],
                    labels=encoded_labels,
                    no_validation=True,
                    stratify=True,
                )

            col, _ = st.columns([1, 1])
            with col:
                st.info("""No test size was selected in the training config page.
                So there's no test set image.""")
                st.code(f"Total training images = {len(X_train)}  \n"
                        f"Total validation images = {len(X_test)}")

        with st.spinner("Saving the test set data ..."):
            images_and_labels = (X_test, y_test, encoded_label_dict)
            with open(self.test_set_pkl_path, "wb") as f:
                pickle.dump(images_and_labels, f)

        # ***************** Preparing tf.data.Dataset *****************
        with st.spinner("Creating TensorFlow dataset ..."):
            if self.has_valid_set:
                train_ds, val_ds, test_ds = self.create_tf_dataset(
                    X_train, y_train, X_test, y_test, X_val, y_val)
            else:
                train_ds, test_ds = self.create_tf_dataset(
                    X_train, y_train, X_test, y_test)

        # ***************** Build model and callbacks *****************
        if not is_resume:
            with st.spinner("Building the model ..."):
                model = self.build_classification_model()
            initial_epoch = 0
            num_epochs = self.training_param['num_epochs']
        else:
            logger.info(f"Loading Model ID {self.training_model_id} "
                        "to resume training ...")
            with st.spinner("Loading trained model to resume training ..."):
                # load the full Keras model instead of weights to easily resume training
                model = self.load_keras_model()
            initial_epoch = session_state.new_training.progress['Epoch']
            num_epochs = initial_epoch + self.training_param['num_epochs']

        progress_placeholder = {}
        progress_placeholder['epoch'] = st.empty()
        progress_placeholder['batch'] = st.empty()
        callbacks = self.create_callbacks(
            train_size=len(y_train), progress_placeholder=progress_placeholder,
            num_epochs=num_epochs)

        # ********************** Train the model **********************
        if self.has_valid_set:
            validation_data = val_ds
        else:
            validation_data = test_ds

        logger.info("Training model...")
        start = time.perf_counter()
        with st.spinner("Training model ..."):
            history = model.fit(
                train_ds,
                validation_data=validation_data,
                initial_epoch=initial_epoch,
                epochs=num_epochs,
                callbacks=callbacks,
                # turn off printing training progress in console
                verbose=0
            )
        time_elapsed = time.perf_counter() - start

        m, s = divmod(time_elapsed, 60)
        m, s = int(m), int(s)
        logger.info(f'Finished training! Took {m}m {s}s')
        st.success(f'Model has finished training! Took **{m}m {s}s**')

        with st.spinner("Loading the model with the best validation loss ..."):
            model = self.load_model_weights(model)

        with st.spinner("Saving the trained TensorFlow model ..."):
            model.save(self.training_path['keras_model_file'])

        # ************************ Evaluation ************************
        with st.spinner("Evaluating on validation set and test set if available ..."):
            if self.has_valid_set:
                (val_loss, val_accuracy) = model.evaluate(val_ds)
                txt_1 = (f"Validation aval_ccuracy: {val_accuracy * 100:.2f}%  \n"
                         f"Validation loss: {val_loss:.4f}")
                st.info(txt_1)

                # show the accuracy on the test set
                (test_loss, test_accuracy) = model.evaluate(test_ds)
                txt_2 = (f"Testing accuracy: {test_accuracy * 100:.2f}%  \n"
                         f"Testing loss: {test_loss:.4f}")
                st.info(txt_2)

                # save the results in a txt file
                with open(self.test_result_txt_path, "w") as f:
                    f.write("  \n".join([txt_1, txt_2]))
            else:
                (test_loss, test_accuracy) = model.evaluate(test_ds)
                txt_1 = (f"Validation accuracy: {test_accuracy * 100:.2f}%  \n"
                         f"Validation loss: {test_loss:.4f}")
                with open(self.test_result_txt_path, "w") as f:
                    f.write(txt_1)

        # show a nicely formatted classification report
        with st.spinner("Making predictions of the test set ..."):
            pred_proba = model.predict(test_ds)
            preds = np.argmax(pred_proba, axis=-1)
            y_true = np.concatenate([y for x, y in test_ds], axis=0)
            unique_labels = [str(encoded_label_dict[label])
                             for label in np.unique(y_true)]

        with st.spinner("Generating classification report ..."):
            classif_report = classification_report(
                y_true, preds, target_names=unique_labels
            )
            st.subheader("Classification report")
            st.text(classif_report)

            # append to the test results
            with open(self.test_result_txt_path, "a") as f:
                header_txt = "  \nClassification report:"
                f.write("  \n".join([header_txt, classif_report]))

        with st.spinner("Creating confusion matrix ..."):
            cm = confusion_matrix(y_true, preds)

            fig = plt.figure()
            ax = sns.heatmap(
                cm,
                cmap="Blues",
                annot=True,
                fmt="d",
                cbar=False,
            )
            plt.title("Confusion Matrix", size=12, fontfamily="serif")
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            # save the figure to reuse later
            plt.savefig(str(self.confusion_matrix_path),
                        bbox_inches="tight")
            st.pyplot(fig)

        # remove the model weights file, which is basically just used for loading
        # the best weights with the lowest val_loss. Decided to just use the
        # full model h5 file to make things easier to resume training.
        weights_path = self.training_path['model_weights_file']
        logger.debug(f"Removing unused model_weights file: {weights_path}")
        os.remove(weights_path)

    def run_classification_eval(self):
        # load back the best model
        model = self.load_keras_model()

        # show the evaluation results stored during training
        with open(self.test_result_txt_path) as f:
            result_txt = f.read()
        st.subheader("Evaluation result on validation set and "
                     "test set if available")
        st.text(result_txt)

        image = plt.imread(self.confusion_matrix_path)
        st.image(image)
        st.markdown("___")

        # ************* Show predictions on test set images *************
        options_col, _ = st.columns([1, 1])
        prev_btn_col_1, next_btn_col_1, _ = st.columns([1, 1, 3])
        image_col, image_col_2 = st.columns([1, 1])
        prev_btn_col_2, next_btn_col_2, _ = st.columns([1, 1, 3])

        if 'start_idx' not in session_state:
            # to keep track of the image index to show
            session_state['start_idx'] = 0

        @st.experimental_memo
        def get_test_images_labels():
            with st.spinner("Getting images and labels ..."):
                with open(self.test_set_pkl_path, 'rb') as f:
                    X_test, y_test, encoded_label_dict = pickle.load(f)
                return X_test, y_test, encoded_label_dict

        X_test, y_test, encoded_label_dict = get_test_images_labels()

        with options_col:
            st.header("Prediction results on test set")

            def reset_start_idx():
                session_state['start_idx'] = 0

            st.number_input(
                "Number of samples to show:",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                format='%d',
                key='n_samples',
                help="Number of samples to run detection and display results.",
                on_change=reset_start_idx
            )

            st.info(f"**Total test set images**: {len(X_test)}")

        n_samples = int(session_state['n_samples'])
        start_idx = session_state['start_idx']
        st.write(f"{start_idx = }")
        st.write(f"{start_idx + n_samples = }")
        current_image_paths = X_test[start_idx: start_idx + n_samples]
        current_labels = y_test[start_idx: start_idx + n_samples]

        def previous_samples():
            if session_state['start_idx'] > 0:
                session_state['start_idx'] -= n_samples

        def next_samples():
            max_start = len(X_test) - n_samples
            if session_state['start_idx'] < max_start:
                session_state['start_idx'] += n_samples

        with prev_btn_col_1:
            st.button('⏮️ Previous samples', key='btn_prev_images_1',
                      on_click=previous_samples)
        with next_btn_col_1:
            st.button('Next samples ⏭️', key='btn_next_images_1',
                      on_click=next_samples)

        logger.info(f"Detecting from the test set images: {start_idx}"
                    f" to {start_idx + n_samples} ...")
        image_cols = cycle((image_col, image_col_2))
        with st.spinner("Running classifications ..."):
            for p, label, col in zip(current_image_paths, current_labels, image_cols):
                filename = os.path.basename(p)

                img = load_image_into_numpy_array(str(p))
                image_size = self.training_param["image_size"]
                resized_img = cv2.resize(img, (image_size, image_size))
                resized_img = np.expand_dims(resized_img, axis=0)

                pred = model.predict(resized_img)
                y_pred = np.argmax(pred, axis=-1)[0]
                pred_classname = encoded_label_dict[y_pred]
                true_classname = encoded_label_dict[label]

                caption = (f"{filename}; "
                           f"Actual: {true_classname}; "
                           f"Predicted: {pred_classname}")

                with col:
                    st.image(img, caption=caption)

        with prev_btn_col_2:
            st.button('⏮️ Previous samples', key='btn_prev_images_2',
                      on_click=previous_samples)
        with next_btn_col_2:
            st.button('Next samples ⏭️', key='btn_next_images_2',
                      on_click=next_samples)

    def export_keras_model(self):
        paths = self.training_path

        # copy first before tarring the export folder
        dest_path = paths['export'] / paths['keras_model_file'].name
        if not dest_path.exists():
            shutil.copy2(paths['keras_model_file'], dest_path)

        # just create a tarfile for the user to download
        with st.spinner("Creating model tarfile to download ..."):
            os.chdir(paths['export'].parent)
            export_foldername = paths['export'].name
            model_tarfile_name = paths['model_tarfile'].name
            # tarring the "export" folder within its parent folder using `tar` command
            run_command(
                f"tar -czf {model_tarfile_name} "
                f"{export_foldername}")
        # and then change back to root dir
        chdir_root()
        logger.debug("Keras model tarfile created at: "
                     f"{paths['model_tarfile']}")
