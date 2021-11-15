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
import sys
import shutil
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np

import pandas as pd
import wget
import streamlit as st
from streamlit import session_state

import tensorflow as tf
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

from sklearn.model_selection import train_test_split
from imutils.paths import list_images

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

# >>>> User-defined Modules >>>>
from core.utils.log import logger  # logger
from data_manager.database_manager import init_connection
from project.project_management import Project
from training.training_management import Training
from training.labelmap_management import Framework, Labels
from path_desc import (TFOD_DIR, PRE_TRAINED_MODEL_DIR,
                       USER_DEEP_LEARNING_MODEL_UPLOAD_DIR, TFOD_MODELS_TABLE_PATH,
                       CLASSIF_MODELS_NAME_PATH, SEGMENT_MODELS_TABLE_PATH, chdir_root)
from .utils import (run_command, run_command_update_metrics_2,
                    find_tfod_metric, load_image_into_numpy_array, load_labelmap,
                    xml_to_csv)
from .visuals import draw_gt_bbox, draw_tfod_bboxes

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration >>>>

# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])

# *********************PAGINATION**********************


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
            project.id).tolist()
        # with keys: 'train', 'eval', 'test'
        self.partition_ratio: Dict[str, float] = new_training.partition_ratio
        self.dataset_export_path: Path = project.get_export_path()
        # NOTE: Might need these later
        # self.attached_model: Model = new_training.attached_model
        # self.training_model: Model = new_training.training_model
        self.training_param: Dict[str, Any] = new_training.training_param_dict
        self.training_path: Dict[str, Path] = new_training.training_path

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
            val_size (Optional[float]): Size of validation split, only needed if `no_validation` is False. Defaults to 0.0.
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
        image_paths = sorted(list_images(image_dir))
        logger.info(f"Total images = {len(image_paths)}")

        # directory to annotation folder, only change this path when necessary
        if annotation_dir is not None:
            # get the label paths and sort them to align with image paths
            label_paths = sorted(list(annotation_dir.iterdir()))
        else:
            label_paths = []

        # TODO: maybe can add stratification as an option (only works for img classification)
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
    def copy_images(image_paths: Path,
                    dest_dir: Path,
                    data_type: str,
                    label_paths: Optional[Path] = None):
        assert data_type in ("train", "valid", "test")
        image_dest = dest_dir / data_type

        logger.info(f"Copying files from {data_type} dataset to {image_dest}")
        if image_dest.exists():
            # remove the existing images
            shutil.rmtree(image_dest, ignore_errors=False)

        # create new directories
        os.makedirs(image_dest)

        if label_paths:
            for image_path, label_path in zip(image_paths, label_paths):
                # copy the image file and label file to the new directory
                shutil.copy2(image_path, image_dest)
                shutil.copy2(label_path, image_dest)
        else:
            for image_path in image_paths:
                shutil.copy2(image_path, image_dest)

    def train(self, is_resume: bool = False, stdout_output: bool = False):
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
            pass
        elif self.deployment_type == "Semantic Segmentation with Polygons":
            pass

    def evaluate(self):
        logger.info(f"Start evaluation for Training {self.training_id}")
        if self.deployment_type == 'Object Detection with Bounding Boxes':
            self.run_tfod_evaluation()
        elif self.deployment_type == "Image Classification":
            pass
        elif self.deployment_type == "Semantic Segmentation with Polygons":
            pass

    def reset_tfod_progress(self):
        # reset the training progress
        training_progress = {'Step': 0, 'Checkpoint': 0}
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
        # this name is based on `generate_labelmap_file` function
        LABEL_MAP_NAME = 'labelmap.pbtxt'

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
            # this is a temporary path for labelmap file, will move to desired path later
            'LABELMAP': paths['ANNOTATION_PATH'] / LABEL_MAP_NAME,
            'MODEL_TARFILE': self.training_path['model_tarfile'],
        }

        # create all the necessary paths if not exists yet
        for path in paths.values():
            if not os.path.exists(path):
                os.makedirs(path)

        # ************* Generate train & test images in the folder *************
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

        col, _ = st.columns([1, 1])
        with col:
            st.code(f"Total training images = {len(y_train)}  \n"
                    f"Total testing images = {len(y_test)}")

        with st.spinner('Copying images to folder, this may take awhile ...'):
            self.copy_images(X_train,
                             dest_dir=paths['IMAGE_PATH'],
                             data_type="train",
                             label_paths=y_train)
            self.copy_images(X_test,
                             dest_dir=paths['IMAGE_PATH'],
                             data_type="test",
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
                                          # labelmap file resides here
                                          dst=paths['ANNOTATION_PATH'],
                                          framework=Framework.TensorFlow,
                                          deployment_type=self.deployment_type)

        # ******************** Generate TFRecords ********************
        with st.spinner('Generating TFRecords ...'):
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
            elif 'faster_rcnn' in PRETRAINED_MODEL_URL:
                pipeline_config.model.faster_rcnn.num_classes = len(
                    CLASS_NAMES)
            else:
                logger.error("Pretrained model config is not found!")

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

        # *********************** EXPORT MODEL ***********************
        with st.spinner("Exporting model ..."):
            if paths['EXPORT'].exists():
                # remove any existing export directory first
                shutil.rmtree(paths['EXPORT'])

            FREEZE_SCRIPT = paths['APIMODEL_PATH'] / 'research' / \
                'object_detection' / 'exporter_main_v2.py '
            command = (f"python {FREEZE_SCRIPT} "
                       "--input_type=image_tensor "
                       f"--pipeline_config_path={files['PIPELINE_CONFIG']} "
                       f"--trained_checkpoint_dir={paths['MODELS']} "
                       f"--output_directory={paths['EXPORT']}")
            run_command(command, stdout_output)
            st.success('Model is successfully exported!')

            # Also copy the label map file to the exported directory to use for display
            # label names in inference later
            # NOTE: self.training_path['labelmap'] is in export path
            shutil.copy2(files['LABELMAP'], self.training_path['labelmap'])

            # tar the exported model to be used anywhere
            # NOTE: be careful with the second argument of tar -czf command,
            #  if you pass a chain of directories, the directories
            #  will also be included in the tarfile.
            # That's why we `chdir` first and pass only the file/folder names
            os.chdir(paths['EXPORT'].parent)
            run_command(
                f"tar -czf {files['MODEL_TARFILE'].name} {paths['EXPORT'].name}")
            # and then change back to root dir
            chdir_root()

            # after created the tarfile in the current working directory,
            #  then only move to the desired filepath
            # ! You don't need this as the tarfile is in the same directory as the export path
            # shutil.move(files['MODEL_TARFILE'].name,
            #             files['MODEL_TARFILE'])

        logger.info(
            f"CONGRATS YO! Successfully created {files['MODEL_TARFILE']}")

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
                    verbose: bool = False) -> Dict[str, Any]:
        """`detect_fn` is obtained using `load_tfod_model` method"""
        start_t = time.perf_counter()
        # Running the infernce on the image specified in the  image path
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.expand_dims`.
        input_tensor = input_tensor[tf.newaxis, ...]
        # input_tensor = tf.expand_dims(input_tensor, 0)

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
        # print(detections['detection_classes'])
        end_t = time.perf_counter()
        if verbose:
            logger.info(f'Done inference. [{end_t - start_t:.2f} secs]')

        return detections

    def run_tfod_evaluation(self):
        """Due to some ongoing issues with using COCO API for evaluation on Windows
        (refer https://github.com/google/automl/issues/487),
        I decided not to use the evaluation script for TFOD. Thus, the manual 
        evaluation here only shows some output images with bounding boxes"""
        # get the required paths
        paths = self.training_path
        with st.spinner("Loading model ... This might take awhile ..."):
            detect_fn = self.load_tfod_model()
        category_index = load_labelmap(paths['labelmap'])

        # **************** SHOW SOME IMAGES FOR EVALUATION ****************
        options_col, _ = st.columns([1, 1])
        prev_btn_col_1, next_btn_col_1, _ = st.columns([1, 1, 3])
        true_img_col, pred_img_col = st.columns([1, 1])
        prev_btn_col_2, next_btn_col_2, _ = st.columns([1, 1, 3])

        if 'start_idx' not in session_state:
            # to keep track of the image index to show
            session_state['start_idx'] = 0

        @st.experimental_memo
        def get_test_images_annotations():
            with st.spinner("Getting images and annotations ..."):
                test_data_dir = paths['images'] / 'test'
                test_img_paths = list(list_images(test_data_dir))
                # get the ground truth bounding box data from XML files
                gt_xml_df = xml_to_csv(str(test_data_dir))
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
            for i, p in enumerate(current_image_paths):
                # current_img_idx = start_idx + i + 1
                img = load_image_into_numpy_array(str(p))

                filename = os.path.basename(p)
                gt_img_boxes = gt_xml_df.loc[
                    gt_xml_df['filename'] == filename, 'xmin': 'ymax'
                ].values

                detections = self.tfod_detect(detect_fn, img, verbose=True)
                img_with_detections = draw_tfod_bboxes(
                    detections, img, category_index,
                    session_state.conf_threshold)

                with true_img_col:
                    st.image(draw_gt_bbox(img, gt_img_boxes),
                             caption=f'Ground Truth: {filename}')
                with pred_img_col:
                    st.image(img_with_detections,
                             caption=f'Prediction: {filename}')

        with prev_btn_col_2:
            st.button('⏮️ Previous samples', key='btn_prev_images_2',
                      on_click=previous_samples)
        with next_btn_col_2:
            st.button('Next samples ⏭️', key='btn_next_images_2',
                      on_click=next_samples)

    def run_classification_training(self):
        pass