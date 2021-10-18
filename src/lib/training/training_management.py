"""
Title: Training Management
Date: 23/7/2021
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

from copy import deepcopy
import json
import sys
import traceback
from collections import namedtuple
from enum import IntEnum
from math import ceil, floor
from pathlib import Path
from time import sleep
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union
from dataclasses import asdict, dataclass, field

import pandas as pd
import project
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state


# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

# >>>> User-defined Modules >>>>
from core.utils.file_handler import create_folder_if_not_exist
from core.utils.form_manager import (check_if_exists, check_if_field_empty,
                                     reset_page_attributes)
from core.utils.helper import (NetChange, datetime_formatter, find_net_change,
                               get_directory_name, join_string)
from core.utils.code_generator import get_random_string
from core.utils.log import logger  # logger
from data_manager.database_manager import (db_fetchall, db_fetchone,
                                           db_no_fetch, init_connection)
from deployment.deployment_management import Deployment, DeploymentType
from project.project_management import Project
from training.model_management import BaseModel, Model, NewModel

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration >>>>

# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])

# *********************PAGINATION**********************


class TrainingPagination(IntEnum):
    Dashboard = 0
    New = 1
    Existing = 2
    NewModel = 3

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return TrainingPagination[s]
        except KeyError:
            raise ValueError()


class NewTrainingPagination(IntEnum):
    InfoDataset = 0
    Model = 1
    TrainingConfig = 2
    AugmentationConfig = 3
    Training = 4

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return NewTrainingPagination[s]
        except KeyError:
            raise ValueError()


# NOTE KIV
PROGRESS_TAGS = {
    DeploymentType.Image_Classification: ['Steps'],
    DeploymentType.OD: ['Checkpoint', 'Steps'],
    DeploymentType.Instance: ['Checkpoint', 'Steps'],
    DeploymentType.Semantic: ['Checkpoint', 'Steps']
}


class NewTrainingSubmissionHandlers(NamedTuple):
    insert: Callable[..., Any]
    update: Callable[..., Any]
    context: Dict
    name_key: str


class DatasetPath(NamedTuple):
    train: Path
    eval: Path
    test: Path

# <<<< Variable Declaration <<<<


# >>>> TODO >>>>
ACTIVATION_FUNCTION = ['RELU_6', 'sigmoid']
OPTIMIZER = []
CLASSIFICATION_LOSS = []


class TrainingParam:
    def __init__(self) -> None:
        self.num_classes: int = 1
        self.batch_size: int = 2
        self.learning_rate: Union[int, float] = 0.0008
        self.num_steps: int = 50000
        self.warmup_steps = 1000
        self.warmup_learning_rate = 0.00001
        self.shuffle_buffer_size = 2048
        self.activation_function: str = 'RELU_6'
        self.optimizer: str = ''
        self.classification_loss: str = 'weighted_sigmoid_focal'
        self.training_param_optional: List = []


@dataclass(eq=False, order=False)
class AugmentationConfig:
    # the interface mode used in the augmentation config page (either "Simple" or "Professional")
    interface_type: str = "Simple"
    # These three attributes are used only for object detection
    min_area: int = None
    min_visibility: float = None
    train_size: int = None
    # this augmentations dictionary is directly used for Albumentations for creating transforms
    # Putting this augmentations attribute at the bottom to display the
    #  augmentation config nicely in the run_training_page
    augmentations: Dict[str, Any] = field(default_factory=dict)

    def __len__(self):
        return len(self.augmentations)

    def exists(self) -> bool:
        """Check if any augmentations have been chosen and submitted for this instance."""
        if len(self.augmentations) > 0:
            return True
        return False

    def reset(self):
        self.augmentations = {}
        self.min_area = None
        self.min_visibility = None
        self.train_size = None

    def to_dict(self) -> Dict[str, Any]:
        # return asdict(self)
        return deepcopy(self.__dict__)  # this is faster than asdict


class BaseTraining:
    def __init__(self, training_id, project: Project) -> None:
        self.id: Union[str, int] = training_id
        self.name: str = ''
        self.desc: str = ''
        # NOTE: Might use this, depends
        # self.training_param: Optional[TrainingParam] = TrainingParam()
        # TODO
        # self.augmentation: Optional[AugmentationConfig] = AugmentationConfig()
        self.project_id: int = project.id
        self.project_path = project.project_path
        self.model_id: int = None
        self.attached_model: Model = None
        self.project_model: Model = None
        self.pre_trained_model_id: Optional[int] = None
        self.deployment_type: str = project.deployment_type
        self.model_path: str = ''
        self.framework: str = ''
        self.partition_ratio: Dict = {
            'train': 0.8,
            'eval': 0.2,
            'test': 0
        }  # UPDATED
        self.partition_size: Dict = {
            'train': 0,
            'eval': 0,
            'test': 0
        }
        # list of dataset names
        self.dataset_chosen: List[str] = []
        self.training_param_dict: Dict = {}
        # note that for object detection, this will also have the following keys:
        #  `min_area`, `min_visibility` and `train_size`
        self.augmentation_config: AugmentationConfig()
        self.is_started: bool = False
        self.progress: Dict = {}
        # currently training_path is created using `property` in the Training class
        # self.training_path: Dict[str, Path] = {
        #     'ROOT': None,
        #     'annotations': None,
        #     'images': None,
        #     'export': None,
        #     'models': None,
        #     'model_tarfile': None,
        #     'labelmap': None,
        #     'tensorboard_logdir': None
        # }
        # this tells whether the data has already been stored in dataset,
        # to be able to tell the submission forms whether we are inserting or updating the DB
        self.has_submitted: Dict = {
            NewTrainingPagination.InfoDataset: False,
            NewTrainingPagination.Model: False,
            NewTrainingPagination.TrainingConfig: False,
            NewTrainingPagination.AugmentationConfig: False
        }

    @staticmethod
    def get_training_path(project_path: Path, training_name: str) -> Path:
        """Get training path from training name and project name


        Args:
            project_path (Path): project_path from Project object class
            training_name (str): Name of Training

        Returns:
            Path: Path-like object for Training path
        """
        directory_name = get_directory_name(
            training_name)  # change name to lowercase
        # join directory name with '-' dash

        project_path = Path(project_path)  # make sure it is Path() object

        training_path = project_path / 'training' / str(directory_name)
        logger.info(f"Training Path: {str(training_path)}")

        return training_path

    def calc_total_dataset_size(self, dataset_chosen: List, dataset_dict: Dict) -> int:
        """Calculate the total dataset size for the current training configuration

        Args:
            dataset_dict (Dict): Dict of datasets attached to project

        Returns:
            int: Total size of the chosen datasets
        """

        total_dataset_size = 0
        for dataset in dataset_chosen:
            # Get dataset namedtuple from dataset_dict
            dataset_info = dataset_dict.get(dataset)
            # Obtain 'Dataset_Size' attribute from namedtuple
            dataset_size = dataset_info.Dataset_Size
            total_dataset_size += dataset_size

        return total_dataset_size

    def calc_dataset_partition_size(self, dataset_chosen: List, dataset_dict: Dict):
        """Calculate partition size of dataset for training

        Args:
            dataset_chosen (List): List of Dataset chosen for Training
            dataset_dict (Dict): Dictionary of Dataset details from Project() class object
        """

        if dataset_chosen:
            total_dataset_size = self.calc_total_dataset_size(
                dataset_chosen, dataset_dict)

            self.partition_size['test'] = floor(
                self.partition_ratio['test'] * total_dataset_size)
            num_train_eval = total_dataset_size - self.partition_size['test']
            self.partition_size['train'] = ceil(num_train_eval * (self.partition_ratio['train']) / (
                self.partition_ratio['train'] + self.partition_ratio['eval']))
            self.partition_size['eval'] = num_train_eval - \
                self.partition_size['train']

    def update_dataset_chosen(self, submitted_dataset_chosen: List, dataset_dict: Dict):
        """ Update the training_dataset table in the Database 

        Args:
            submitted_dataset_chosen (List): List of updated Dataset chosen for Training
            dataset_dict (Dict): Dictionary of Dataset details from Project() class object
        """
        # find net change in dataset_chosen List
        appended_elements, append_flag = find_net_change(
            self.dataset_chosen, submitted_dataset_chosen)

        self.dataset_chosen = submitted_dataset_chosen
        if append_flag == NetChange.Addition:
            self.insert_training_dataset(appended_elements)

        elif append_flag == NetChange.Removal:
            self.remove_training_dataset(appended_elements, dataset_dict)

        else:
            logger.info(
                f"There are no change in the dataset chosen {self.dataset_chosen}")

    def remove_training_dataset(self, removed_dataset: List, dataset_dict: Dict):
        """Remove dataset from training_dataset table in the Database

        Args:
            removed_dataset (List): List of removed dataset
            dataset_dict (Dict): Dictionary of Dataset details from Project() class object


        """
        #  remove row from training_dataset
        remove_training_dataset_SQL = """
                DELETE FROM public.training_dataset
                WHERE training_id = %s
                    AND dataset_id = (
                        SELECT
                            id
                        FROM
                            public.dataset d
                        WHERE
                            d.name = %s)
                RETURNING
                    dataset_id;
                        """
        for dataset in removed_dataset:
            remove_training_dataset_vars = [self.id, dataset]

            try:
                query_return = db_fetchone(
                    remove_training_dataset_SQL, conn, remove_training_dataset_vars).dataset_id

                true_dataset_id = dataset_dict[dataset].ID
                assert query_return == true_dataset_id, f"Removed wrong dataset of ID {query_return}, should be {true_dataset_id}"
                logger.info(f"Removed Training Dataset {dataset}")

            except Exception as e:
                logger.error(f"{e}")

    def insert_training_dataset(self, added_dataset: List):
        """Insert Dataset into training_dataset table

        Args:
            added_dataset (List): Dataset chosen for training
        """
        # submission handler for insertion of rows into training dataset table

        insert_training_dataset_SQL = """
                    INSERT INTO public.training_dataset (
                        training_id
                        , dataset_id)
                    VALUES (
                        %s
                        , (
                            SELECT
                                d.id
                            FROM
                                public.dataset d
                            WHERE
                                d.name = %s))
                    ON CONFLICT ON CONSTRAINT training_dataset_pkey
                        DO NOTHING;
                    """
        for dataset in added_dataset:
            insert_training_dataset_vars = [self.id, dataset]
            db_no_fetch(insert_training_dataset_SQL, conn,
                        insert_training_dataset_vars)

            logger.info(f"Inserted Training Dataset {dataset}")

    def update_training_info(self) -> bool:
        """Update Training Name, Description, and Partition Size into the database

        Returns:
            bool: True is successful, otherwise False
        """
        update_training_info_SQL = """
            UPDATE
                public.training
            SET
                name = %s
                , description = %s
                , partition_ratio = %s::JSONB
            WHERE
                id = %s
            RETURNING
                id;
        
                                """
        partition_ratio_json = json.dumps(self.partition_ratio, indent=4)
        update_training_info_vars = [self.name,
                                     self.desc, partition_ratio_json, self.id]

        query_return = db_fetchone(update_training_info_SQL,
                                   conn,
                                   update_training_info_vars).id

        try:
            assert self.id == query_return, f'Updated wrong Training of ID {query_return}, which should be {self.id}'
            logger.info(
                f"Successfully updated New Training Name, Desc and Partition Size for {self.id} ")
            return True

        except Exception as e:
            logger.error(
                f"{e}: Failed to update New Training Name, Desc and Partition Size for {self.id}")
            return False

    def insert_training_info(self) -> bool:
        """Insert Training Name, Description and Partition Size into Training table

        Returns:
            bool: True if successfully inserted into the database
        """

        insert_training_info_SQL = """
                INSERT INTO public.training (
                    name
                    , description
                    , partition_ratio,
                    project_id)
                VALUES (
                    %s
                    , %s
                    , %s::JSONB,
                    %s)
                ON CONFLICT (
                    name)
                    DO UPDATE SET
                        description = %s
                        , partition_ratio = %s::JSONB
                    RETURNING
                        id;
                                    """

        partition_ratio_json = json.dumps(self.partition_ratio, indent=4)
        insert_training_info_vars = [self.name, self.desc, partition_ratio_json,
                                     self.project_id, self.desc, partition_ratio_json]

        try:
            query_return = db_fetchone(insert_training_info_SQL,
                                       conn,
                                       insert_training_info_vars).id
            self.id = query_return
            logger.info(
                f"Successfully load New Training Name, Desc and Partition Size for {self.id} ")
            return True

        except TypeError as e:
            logger.error(
                f"{e}: Failed to load New Training Name, Desc and Partition Size for {self.id}")
            return False

    def insert_project_training(self):
        insert_project_training_SQL = """
            INSERT INTO public.project_training (
                project_id
                , training_id)
            VALUES (
                %s
                , %s)
            ON CONFLICT ON CONSTRAINT project_training_pkey
                DO NOTHING;
        
            """
        insert_project_training_vars = [self.project_id, self.id]
        db_no_fetch(insert_project_training_SQL, conn,
                    insert_project_training_vars)

    def update_training_info_dataset(self,
                                     submitted_dataset_chosen: List,
                                     dataset_dict: Dict) -> bool:
        """Update Training Info and Dataset 

        Args:
            submitted_dataset_chosen (List): Modified List of Dataset Chosen
            dataset_dict (Dict): Dictionary of dataset information

        Returns:
            bool: True upon successful update, False otherwise
        """

        try:

            assert self.partition_ratio['eval'] > 0.0, "Dataset Evaluation Ratio needs to be > 0"

            self.update_training_info()

            self.update_dataset_chosen(submitted_dataset_chosen=submitted_dataset_chosen,
                                       dataset_dict=dataset_dict)

            return True

        except Exception as e:
            logger.error(f"{e}")
            traceback.print_exc()

            return False

    def update_training_attached_model(self, attached_model_id: int, training_model_id: int) -> bool:
        # update training table with attached model id and training model id

        insert_training_attached_SQL = """

            UPDATE
                public.training
            SET
                training_model_id = %s
                , attached_model_id = %s
            WHERE
                id = %s;
                    """

        insert_training_attached_vars = [
            training_model_id, attached_model_id, self.id]

        try:
            db_no_fetch(insert_training_attached_SQL, conn,
                        insert_training_attached_vars)
            return True

        except Exception as e:
            logger.error(f"At update training_attached: {e}")
            return False

    def update_training_param(self, training_param: Dict[str, Any]) -> bool:
        # Maybe can try using the TrainingParam class, but seems like not necessary
        self.training_param_dict = training_param

        # required for storing JSONB format
        training_param_json = json.dumps(training_param)
        sql_query = """
        UPDATE
            public.training
        SET
            training_param = %s::JSONB
        WHERE
            id = %s
        """
        query_vars = [training_param_json, self.id]
        try:
            db_no_fetch(sql_query, conn, query_vars)
            return True
        except Exception as e:
            logger.error(f"Update training param failed: {e}")
            return False

    def update_augment_config(self, augmentation_config: AugmentationConfig) -> bool:
        self.augmentation_config = augmentation_config

        # required for storing JSONB format
        augmentation_json = json.dumps(augmentation_config.to_dict())
        sql_query = """
        UPDATE
            public.training
        SET
            augmentation = %s::JSONB
        WHERE
            id = %s
        """
        query_vars = [augmentation_json, self.id]
        try:
            db_no_fetch(sql_query, conn, query_vars)
            return True
        except Exception as e:
            logger.error(f"Update training param failed: {e}")
            return False

    def has_augmentation(self) -> bool:
        """Check if any augmentations have been chosen and submitted for this instance."""
        return self.augmentation_config.exists()


class NewTraining(BaseTraining):
    def __init__(self, training_id, project: Project) -> None:
        super().__init__(training_id, project)

        self.model_selected: NewModel = None  # DEPRECATED
        self.attached_model: Model = None
        self.training_model: NewModel = NewModel()
    # TODO *************************************

    def __repr__(self):
        # to add more later
        return (f"NewTraining(training_id={self.id}, project_id={self.project_id}, "
                f"attached_model={self.attached_model}, training_model={self.training_model})")

    @staticmethod
    # Wrapper for check_if_exists function from form_manager.py
    def check_if_exists(context: Dict[str, Any], conn) -> bool:
        table = 'public.training'

        exists_flag = check_if_exists(
            table, context['column_name'], context['value'], conn)

        # return True if exists
        return exists_flag

    # Wrapper for check_if_exists function from form_manager.py
    def check_if_field_empty(self, context: Dict,
                             field_placeholder: Dict,
                             name_key: str
                             ) -> bool:
        """Check if Compulsory fields are filled and Unique information not 
        duplicated in the database

        Args:
            context (Dict): Dictionary with widget name as key and widget value as value
            field_placeholder (Dict): Dictionary with st.empty() key as key and st.empty() object as value. 
            *Key has same name as its respective widget

            name_key (str): Key of Database row name. Used to obtain value from 'context' Dictionary.
            *Pass 'None' is not required to check row exists

        Returns:
            bool: True if NOT EMPTY + NOT EXISTS, False otherwise.
        """
        check_if_exists = self.check_if_exists
        empty_fields = check_if_field_empty(
            context, field_placeholder, name_key, check_if_exists)

        # True if not empty, False otherwise
        return empty_fields

    def insert_training_info_dataset(self) -> bool:
        """Create New Training submission

        Returns:
            bool: True upon successful creation, False otherwise
        """

        # submission handler for insertion of Info and Dataset
        try:
            assert self.partition_ratio['eval'] > 0.0, "Dataset Evaluation Ratio needs to be > 0"

            # insert Name, Desc, Partition Size
            self.insert_training_info()
            assert isinstance(
                self.id, int), f"Training ID should be type Int but obtained {type(self.id)} ({self.id})"

            # Insert Project Training
            self.insert_project_training()

            # Insert Training Dataset
            self.insert_training_dataset(added_dataset=self.dataset_chosen)

            # NOTE KIV -> to create training folder BEFORE TRAINING
            # Create Training Folder
            # self.initialise_training_folder()

            return True

        except AssertionError as e:
            logger.error(f'{e}')
            traceback.print_exc()
            st.error(f'{e}')
            return False


# NOTE ******************* DEPRECATED *********************************************
    # def insert_training(self, model: Model, project: Project):

    #     # insert into training table
    #     insert_training_SQL = """
    #         INSERT INTO public.training (
    #             name,
    #             description,
    #             training_param,
    #             augmentation,
    #             pre_trained_model_id,
    #             framework_id,
    #             project_id,
    #             partition_size)
    #         VALUES (
    #             %s,
    #             %s,
    #             %s::jsonb,
    #             %s::jsonb,
    #             (
    #                 SELECT
    #                     pt.id
    #                 FROM
    #                     public.pre_trained_models pt
    #                 WHERE
    #                     pt.name = %s), (
    #                     SELECT
    #                         f.id
    #                     FROM
    #                         public.framework f
    #                     WHERE
    #                         f.name = %s), %s, %s)
    #         RETURNING
    #             id;

    #         """
    #     # convert dictionary into serialised JSON
    #     self.training_param_json = json.dumps(
    #         self.training_param, sort_keys=True, indent=4)
    #     self.augmentation_json = json.dumps(
    #         self.augmentation, sort_keys=True, indent=4)
    #     insert_training_vars = [self.name, self.desc, self.training_param_json, self.augmentation_json,
    #                             self.model_selected, self.framework, self.project_id, self.partition_ratio]
    #     self.training_id = db_fetchone(
    #         insert_training_SQL, conn, insert_training_vars).id

    #     # Insert into project_training table
    #     insert_project_training_SQL = """
    #         INSERT INTO public.project_training (
    #             project_id,
    #             training_id)
    #         VALUES (
    #             %s,
    #             %s);

    #         """
    #     insert_project_training_vars = [self.project_id, self.training_id]
    #     db_no_fetch(insert_project_training_SQL, conn,
    #                 insert_project_training_vars)

    #     # insert into model table
    #     insert_model_SQL = """
    #         INSERT INTO public.models (
    #             name,
    #             model_path,
    #             training_id,
    #             framework_id,
    #             deployment_id)
    #         VALUES (
    #             %s,
    #             %s,
    #             %s,
    #             (
    #                 SELECT
    #                     f.id
    #                 FROM
    #                     public.framework f
    #                 WHERE
    #                     f.name = %s), (
    #                     SELECT
    #                         dt.id
    #                     FROM
    #                         public.deployment_type dt
    #                     WHERE
    #                         dt.name = %s))
    #         RETURNING
    #             id;
    #         """
    #     insert_model_vars = [model.name, model.model_path,
    #                          self.training_id, self.framework, project.deployment_type]
    #     model.id = db_fetchone(insert_model_SQL, conn, insert_model_vars).id

    #     # Insert into training_dataset table
    #     insert_training_dataset_SQL = """
    #         INSERT INTO public.training_dataset (
    #             training_id,
    #             dataset_id)
    #         VALUES (
    #             %s,
    #             (
    #                 SELECT
    #                     id
    #                 FROM
    #                     public.dataset d
    #                 WHERE
    #                     d.name = %s))"""
    #     for dataset in self.dataset_chosen:
    #         insert_training_dataset_vars = [self.training_id, dataset]
    #         db_no_fetch(insert_training_dataset_SQL, conn,
    #                     insert_training_dataset_vars)

    #     return self.id

# TODO #133 Add New Training Reset


    @staticmethod
    def reset_new_training_page():

        new_training_attributes = ["new_training", "new_training_name", "new_training_pagination",
                                   "new_training_desc"]

        reset_page_attributes(new_training_attributes)


class Training(BaseTraining):

    def __init__(self, training_id, project: Project) -> None:
        super().__init__(training_id, project)

        # `query_all_fields` creates self.name, self.desc, self.training_param_dict,
        # self.augmentation_config, self.progress, self.partition_ratio
        # self.training_model_id, self.attached_model_id, self.is_started
        # from `training` table
        self.query_all_fields()
        # creates from `training_dataset` table; could have multiple datasets
        self.dataset_chosen = self.query_dataset_chosen(self.id)
        # creates self.attached_model and self.training_model
        self.get_training_details()
        # NOTE: self.training_path is now created with `property` decorator below
        # self.training_path = self.get_all_training_path()

    def __repr__(self):
        return "<{klass} {attrs}>".format(
            klass=self.__class__.__name__,
            attrs=" ".join("{}={!r}".format(k, v)
                           for k, v in self.__dict__.items()),
        )

    @classmethod
    def from_new_training(cls, new_training: NewTraining, project: Project):
        logger.debug("Converting from NewTraining to Training instance")
        new_instance = cls(new_training.id, project)

        for k, v in new_training.__dict__.items():
            # - this `hasattr` checking can optionally be removed in the future,
            # currently using it here for debugging
            if hasattr(new_instance, k):
                setattr(new_instance, k, v)
            else:
                logger.debug(
                    f'Skipping `{k}` attribute when converting to Training instance')

        return new_instance

    def get_training_details(self):
        if self.attached_model_id and self.training_model_id:
            self.attached_model = Model(model_id=self.attached_model_id)
            self.training_model = Model(model_id=self.training_model_id)
        else:
            self.attached_model = None
            self.training_model = NewModel()

    @property
    def training_path(self) -> Dict[str, Path]:
        # modified from get_all_training_path
        # NOTE: need to exclude file paths from the `initialise_training_folder` method
        # >>>> TRAINING PATH
        paths = {}
        paths['ROOT'] = self.get_training_path(self.project_path, self.name)
        root = paths['ROOT']
        # >>>> MODEL PATH
        # paths['models'] = root / 'models'
        if not self.training_model.name:
            model_dirname = 'temp'
        else:
            # MUST use this to get a nicely formatted directory name
            model_dirname = get_directory_name(self.training_model.name)
        paths['models'] = root / 'models' / model_dirname
        # PARTITIONED DATASET PATH
        paths['images'] = root / 'images'
        # EXPORTED MODEL PATH
        # paths['exported_models'] = root / \
        #     'exported_models'
        paths['export'] = paths['models'] / 'export'
        # model weights only available for image classification and segmentation tasks
        # when using Keras
        paths['model_weights'] = paths['export'] / f"{model_dirname}.h5"
        paths['model_tarfile'] = paths['models'] / \
            f'{model_dirname}.tar.gz'
        # ANNOTATIONS PATH
        paths['annotations'] = root / 'annotations'
        # this filename is based on the `generate_labelmap_file` function
        # NOTE: this file is probably only needed for TF object detection
        paths['labelmap'] = paths['export'] / 'labelmap.pbtxt'
        # the tensorboard logdir is the same with models path but just to be explicit
        paths['tensorboard_logdir'] = paths['models']
        return paths

    @staticmethod
    def query_progress(training_id: int) -> Union[bool, None]:
        # NOTE: this method is not being used for now
        sql_query = """
                SELECT
                    is_started
                FROM
                    public.training
                WHERE
                    id = %s;
        """
        query_vars = [training_id]

        is_started = db_fetchone(sql_query, conn, query_vars)  # return tuple

        if is_started:
            return is_started[0]
        else:
            logger.error(
                f"Training with ID {training_id} does not exists in the Database!!!")
            return None

    def update_progress(self,
                        progress: Dict[str, int],
                        is_started: bool = True,
                        verbose: bool = False):
        # NOTE: progress for TFOD should be {'Step': <int>, 'Checkpoint': <int>}

        self.is_started = is_started
        self.progress = progress

        sql_query = """
                UPDATE
                    public.training
                SET
                    is_started = %s,
                    progress = %s::JSONB
                WHERE
                    id = %s;
        """
        progress_json = json.dumps(progress)
        query_vars = [is_started, progress_json, self.id]

        db_no_fetch(sql_query, conn, query_vars)
        if verbose:
            logger.info(f"Updated progress for Training {self.id} "
                        f"with: '{progress}'")

    def update_metrics(self, result_metrics: Dict[str, Any], verbose: bool = False):
        self.training_model.metrics = result_metrics

        sql_query = """
                UPDATE
                    public.models
                SET
                    metrics = %s::JSONB
                WHERE
                    id = %s;
        """
        metrics_json = json.dumps(result_metrics)
        query_vars = [metrics_json, self.training_model.id]

        db_no_fetch(sql_query, conn, query_vars)
        if verbose:
            logger.info(f"Updated metrics for Model {self.training_model.id} "
                        f"with: '{result_metrics}'")

    def query_all_fields(self) -> NamedTuple:
        """Query fields of current Training
        - name
        - description
        - training_param
        - augmentation
        - is_started
        - progress
        - partition_ratio
        - training_model_id
        - attached_model_id

        Returns:
            NamedTuple: Query results from Training table
        """
        query_all_fields_SQL = """
                SELECT
                    name
                    , description
                    , training_param
                    , augmentation
                    , is_started
                    , progress
                    , partition_ratio
                    , training_model_id
                    , attached_model_id
                FROM
                    public.training
                WHERE
                    id = %s;
        
                             """
        assert (isinstance(self.id, int)
                ), f"Training ID should be type int but {type(self.id)} obtained ({self.id})"

        query_all_fields_vars = [self.id]

        training_field = db_fetchone(
            query_all_fields_SQL, conn, query_all_fields_vars)

        if training_field:
            self.name, self.desc,\
                self.training_param_dict, augmentation_config,\
                self.is_started, self.progress, self.partition_ratio, \
                self.training_model_id, self.attached_model_id = training_field
            # need to convert from Dictionary to the dataclass
            if augmentation_config is not None:
                self.augmentation_config = AugmentationConfig(
                    **augmentation_config)
            else:
                self.augmentation_config = AugmentationConfig()
        else:
            logger.error(
                f"Training with ID {self.id} for Project ID {self.project_id} does not exists in the Database!!!")

            training_field = None

        return training_field

    @staticmethod
    def query_dataset_chosen(training_id: int) -> NamedTuple:
        sql_query = """
                SELECT  d.id   AS id,
                        d.name AS dataset_chosen
                FROM training_dataset t
                        JOIN dataset d ON t.dataset_id = d.id
                WHERE training_id = %s;
        """
        query_vars = [training_id]

        # return List[namedtuple]
        query_return = db_fetchall(sql_query, conn, query_vars)

        if query_return:
            dataset_chosen = [record.dataset_chosen for record in query_return]
            logger.info(
                f"Training of ID {training_id} has selected datasets: {dataset_chosen}")
        else:
            logger.info(
                f"Training of ID {training_id} has not selected any training_dataset yet")

        return dataset_chosen

    @staticmethod
    def query_all_project_training(project_id: int,
                                   deployment_type: Union[str, IntEnum],
                                   return_dict: bool = False,
                                   for_data_table: bool = False,
                                   progress_preprocessing: bool = False
                                   ) -> Union[List[namedtuple], List[dict]]:
        """Query All Trainings bounded to current Project ID

        Args:
            project_id (int): Project ID
            deployment_type (Union[str, IntEnum]): Deployment Type
            return_dict (bool, optional): True if needed for Data Table. Defaults to False.
            for_data_table (bool, optional): Flag for Data Table. Defaults to False.
            progress_preprocessing (bool, optional): True if require to process 'progress' column into human-friendly form. Defaults to False.

        Returns:
            Union[List[namedtuple], List[dict]]: List of Project Training queries from Database
        """

        ID_string = "id" if for_data_table else "ID"
        query_all_project_training_SQL = f"""
                    SELECT
                        t.id AS \"{ID_string}\",
                        t.name AS "Training Name",
                        (
                            SELECT
                                CASE
                                    WHEN t.training_model_id IS NULL THEN '-'
                                    ELSE (
                                        SELECT
                                            m.name
                                        FROM
                                            public.models m
                                        WHERE
                                            m.id = t.training_model_id
                                    )
                                END AS "Model Name"
                        ),
                        (
                            SELECT
                                m.name AS "Base Model Name"
                            FROM
                                public.models m
                            WHERE
                                m.id = t.attached_model_id
                        ),
                        t.is_started AS "Is Started",
                        CASE
                            WHEN t.progress IS NULL THEN \'{{}}\'
                            ELSE t.progress
                        END AS "Progress",
                        /*Give empty JSONB if training progress has not start*/
                        t.updated_at AS "Date/Time"
                    FROM
                        public.project_training pro_train
                        INNER JOIN training t ON t.id = pro_train.training_id
                    WHERE
                        pro_train.project_id = %s;
                                                    """

        query_all_project_training_vars = [project_id]
        logger.info(
            f"Querying Training from database for Project {project_id}")
        all_project_training = []
        try:

            all_project_training_query_return, column_names = db_fetchall(
                query_all_project_training_SQL, conn,
                query_all_project_training_vars, fetch_col_name=True, return_dict=return_dict)

            if progress_preprocessing:
                all_project_training = Training.datetime_progress_preprocessing(
                    all_project_training_query_return, deployment_type, return_dict)

            else:
                all_project_training = datetime_formatter(
                    all_project_training_query_return, return_dict=return_dict)

        except TypeError as e:

            logger.error(f"""{e}: Could not find any training for Project ID {project_id} / \
                            Project ID: Project ID {project_id} does not exists""")

        return all_project_training, column_names

    def initialise_training_folder(self):
        """Create Training Folder
        """
        '''
        training_dir
        |
        |-annotations/
        | |-labelmap
        | |-TFRecords*
        |
        |-exported_models/
        |-dataset
        | |-train/
        | |-evaluation/
        |
        |-models/

        '''
        # *************** GENERATE TRAINING PATHS ***************
        training_paths = self.training_path

        # >>>> CREATE Training directory recursively
        filepath_keys = ('model_tarfile', 'labelmap', 'keras_model', 'model_weights')
        for key, path in training_paths.items():
            if key in filepath_keys:
                # this is a file path and not a directory, thus skipping
                continue
            create_folder_if_not_exist(path)
            logger.info(
                f"Successfully created **{str(path)}**")

        # >>>> ASSERT IF TRAINING DIRECTORY CREATED CORRECTLY
        try:
            assert (len([x for x in training_paths['ROOT'].iterdir() if x.is_dir()]) == (
                len(training_paths) - 1)), f"Failed to create all folders"
            logger.info(
                f"Successfully created **{self.name}** training at {str(training_paths['ROOT'])}")

        except AssertionError as e:

            # Create a list of sub-folders excluding Training Root
            directory_structure = [
                x for x in training_paths.values() if x != training_paths['ROOT']]

            missing_folders = [x for x in training_paths['ROOT'].iterdir(
            ) if x.is_dir() and (x not in directory_structure)]

            logger.error(f"{e}: Missing {missing_folders}")

    # ! Deprecated, use training_path property
    # @staticmethod
    # def get_trained_filepaths(project_path: str,
    #                           training_name: str,
    #                           model_name: str,
    #                           deployment_type: str,
    #                           ) -> Dict[str, Path]:
    #     training_path = BaseTraining.get_training_path(
    #         project_path, training_name)

    #     # TODO for image classification and segmentation
    #     if deployment_type in (DeploymentType.OD, 'Object Detection with Bounding Boxes'):
    #         model_path = training_path / 'models' / model_name
    #         model_export_path = model_path / 'export'
    #         model_tarfile_path = model_path / f'{model_name}.tar.gz'
    #         return {
    #             'training_path': training_path,
    #             'model_path': model_path,
    #             'model_export_path': model_export_path,
    #             'model_tarfile_path': model_tarfile_path
    #         }
    #     elif deployment_type in (DeploymentType.Image_Classification, 'Image Classification'):
    #         pass
    #     elif deployment_type in (DeploymentType.Semantic, 'Semantic Segmentation with Polygons'):
    #         pass
    #     else:
    #         logger.error(f"Error with deployment_type: '{deployment_type}'")


# NOTE ******************* DEPRECATED *********************************************
    # def initialise_training(self, model: Model, project: Project):
    #     '''
    #     training_dir
    #     |
    #     |-annotations/
    #     | |-labelmap
    #     | |-TFRecords*
    #     |
    #     |-exported_models/
    #     |-dataset
    #     | |-train/
    #     | |-evaluation/
    #     |
    #     |-models/

    #     '''
    #     directory_name = get_directory_name(self.name)

    #     self.training_path = project.project_path / \
    #         str(directory_name)  # use training name
    #     self.main_model_path = self.training_path / 'models'
    #     self.dataset_path = self.training_path / 'dataset'
    #     self.exported_model_path = self.training_path / 'exported_models'
    #     self.annotations_path = self.training_path / 'annotations'
    #     directory_pipeline = [self.annotations_path, self.exported_model_path, self.main_model_path,
    #                           self.dataset_path]

    #     # CREATE Training directory recursively
    #     for dir in directory_pipeline:
    #         create_folder_if_not_exist(dir)
    #         logger.info(
    #             f"Successfully created **{str(dir)}**")

    #     logger.info(
    #         f"Successfully created **{self.name}** training at {str(self.training_path)}")

    #     # Entry into DB for training,project_training,model,model_training tables
    #     if self.insert_training():

    #         logger.info(
    #             f"Successfully stored **{self.name}** traiing information in database")
    #         return True

    #     else:
    #         logger.error(
    #             f"Failed to stored **{self.name}** training information in database")
    #         return False
# NOTE ******************* DEPRECATED *********************************************

    @staticmethod
    def datetime_progress_preprocessing(all_project_training: Union[List[NamedTuple], List[Dict]],
                                        deployment_type: Union[str, IntEnum],
                                        return_dict: bool = False
                                        ) -> Union[List[NamedTuple], List[Dict]]:
        """Preprocess Date/Time and Progress column

        Args:
            all_project_training (Union[List[namedtuple], List[Dict]]): All Project Training query results
            return_dict (bool, optional): True if all_project_training is type Dict. Defaults to False.

        Returns:
            Union[List[namedtuple], List[Dict]]: Formatted list of all_project_training
        """
        logger.info(f"Formatting Date/Time and Progress column......")

        deployment_type = Deployment.get_deployment_type(
            deployment_type)  # Make sure it is IntEnum

        # get progress tags based on Deployment Type
        progress_tag = PROGRESS_TAGS[deployment_type]

        formatted_all_project_training = []
        for project_training in all_project_training:
            # convert datetime with TZ to (2021-07-30 12:12:12) format
            if return_dict:
                converted_datetime = project_training["Date/Time"].strftime(
                    '%Y-%m-%d %H:%M:%S')
                project_training["Date/Time"] = converted_datetime

                if project_training['Is Started'] == True:
                    # Preprocess
                    progress_value = []

                    for tag in progress_tag:
                        # for k, v in project_training["Progress"].items():

                        # if k in progress_tag:

                        v = project_training["Progress"].get(
                            tag)if project_training["Progress"].get(
                            tag) is not None else '-'
                        progress_value.append(str(v))

                    progress_row = join_string(progress_value, ' / ')

                    project_training["Progress"] = progress_row
                else:
                    project_training["Progress"] = '-'

            else:
                converted_datetime = project_training.Date_Time.strftime(
                    '%Y-%m-%d %H:%M:%S')

                project_training = project_training._replace(
                    Date_Time=converted_datetime)

                if project_training.Is_Started == True:
                    # Preprocess
                    progress_value = []
                    # for k, v in project_training.Progress.items():
                    for tag in progress_tag:

                        v = project_training.Progress.get(
                            tag)if project_training.Progress.get(
                            tag) is not None else '-'
                        progress_value.append(str(v))

                    progress_row = join_string(progress_value, ' / ')

                    project_training = project_training._replace(
                        Progress=progress_row)

                else:
                    project_training = project_training._replace(
                        Progress='-')

            formatted_all_project_training.append(project_training)

        return formatted_all_project_training

    def reset_training_progress(self, progress: Optional[Dict[str, int]] = None):
        if not progress:
            # can pass in a Dictionary with {'Step': 0, 'Checkpoint': 0} instead
            progress = {}
        self.update_progress(progress, is_started=False, verbose=True)

        # reset the training result metrics
        self.update_metrics({})

    def clone_training_session(self):
        """
        Clone the current training session to allow users to have faster access to training,
        while retaining all the configuration selected for the current training session.
        After this, the user should be moved to the training info page for modification.
        Also, do not forget to reset all the `has_submmitted` values to False to allow 
        the user to seamlessly navigate through the training & model selection forms.
        """
        # creating a temporary training session's name
        new_training_name = '-'.join(
            [self.name, get_random_string(length=5)])
        context = {'column_name': 'name', 'value': new_training_name}
        # just in case the database has the same training session's name
        while NewTraining.check_if_exists(context, conn):
            new_training_name = '-'.join(
                [self.name, get_random_string(length=5)])
            context = {'column_name': 'name', 'value': new_training_name}

        self.name = new_training_name

        # creating a temporary custom model name
        new_model_name = '-'.join(
            [self.training_model.name, get_random_string(length=5)])
        context = {'column_name': 'name', 'value': new_model_name}
        # just in case the database has the same custom model's name
        while BaseModel.check_if_exists(context, conn):
            new_model_name = '-'.join(
                [self.training_model.name, get_random_string(length=5)])
            context = {'column_name': 'name', 'value': new_model_name}

        self.training_model.name = new_model_name

        # insert the clone into the 'training' table and
        # update self.id with the returned ID
        insert_training_info_SQL = """
                INSERT INTO public.training (
                    name,
                    description,
                    attached_model_id,
                    training_param,
                    augmentation,
                    partition_ratio,
                    is_started,
                    progress,
                    project_id)
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s::JSONB,
                    %s::JSONB,
                    %s::JSONB,
                    %s,
                    %s::JSONB,
                    %s)
                RETURNING
                    id;
        """

        partition_ratio_json = json.dumps(self.partition_ratio)
        training_param_json = json.dumps(self.training_param_dict)
        augment_param_json = json.dumps(self.augmentation_config.to_dict())
        progress_json = json.dumps(self.progress)
        # CARE self.training_model.id is NOT THE NEW ONE YET
        insert_training_info_vars = [self.name, self.desc, self.attached_model.id,
                                     training_param_json, augment_param_json, partition_ratio_json,
                                     self.is_started, progress_json, self.project_id]

        try:
            query_return = db_fetchone(insert_training_info_SQL,
                                       conn,
                                       insert_training_info_vars).id
            # NOTE: this will also update self.id with the new id returned from DB
            self.id = query_return
            logger.info(
                f"Successfully load New Training Name, Desc and Partition Size for {self.id} ")
            # return True
        except TypeError as e:
            logger.error(
                f"{e}: Failed to load New Training Name, Desc and Partition Size for {self.id}")
            # return False

        # also update the training ID in the `training_model` attribute
        self.training_model.training_id = self.id

        # Insert as a new Project Training
        self.insert_project_training()

        # Insert Training Dataset
        self.insert_training_dataset(added_dataset=self.dataset_chosen)

        # insert as a new training model and
        # update the model ID with the ID returned from DB
        self.training_model.insert_new_model(
            model_type=self.training_model.model_type)

        self.update_training_attached_model(
            self.attached_model.id,
            self.training_model.id
        )

        # finally, reset the clone's training progress and training model's metrics
        # as we allow the user to use it to train a new model
        self.reset_training_progress()

    @staticmethod
    def reset_training_page():
        training_attributes = ["training", "training_pagination", "labelling_pagination",
                               "training_param_dict", "new_training", "trainer", "start_idx",
                               "augmentation_config"
                               ]
        # this might be required to avoid issues with caching model-related variables
        # NOTE: this method has moved from `caching` to `legacy_caching` module in v0.89
        # https://discuss.streamlit.io/t/button-to-clear-cache-and-rerun/3928/12
        logger.debug("Clearing all existing Streamlit cache")
        st.legacy_caching.clear_cache()

        reset_page_attributes(training_attributes)


def main():
    print("Hi")


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
