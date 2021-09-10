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

import json
import sys
import traceback
from collections import namedtuple
from enum import IntEnum
from math import ceil, floor
from pathlib import Path
from time import sleep
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

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
from core.utils.log import log_error, log_info  # logger
from data_manager.database_manager import (db_fetchall, db_fetchone,
                                           db_no_fetch, init_connection)
from deployment.deployment_management import Deployment, DeploymentType
from project.project_management import Project
from training.model_management import Model, NewModel

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
        self.optimizer: str = None
        self.classification_loss: str = 'weighted_sigmoid_focal'
        self.training_param_optional: List = []


# >>>> TODO >>>>


class Augmentation:
    def __init__(self) -> None:
        pass


class BaseTraining:
    def __init__(self, training_id, project: Project) -> None:
        self.id: Union[str, int] = training_id
        self.name: str = None
        self.desc: str = None
        self.training_param: Optional[TrainingParam] = TrainingParam()
        self.augmentation: Optional[Augmentation] = Augmentation()  # TODO
        self.project_id: int = project.id
        self.project_path = project.project_path
        self.model_id: int = None
        self.attached_model: Model = None
        self.project_model: Model = None
        self.pre_trained_model_id: Optional[int] = None
        self.deployment_type: str = project.deployment_type
        self.model_path: str = None
        self.framework: str = None
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
        self.dataset_chosen: List = []
        self.training_param_dict: Dict = {}
        self.augmentation_dict: Dict = {}
        self.is_started: bool = False
        self.progress: Dict = {}
        self.training_path: Dict = {
            'ROOT': None,
            'annotations': None,
            'dataset': None,
            'exported_models': None,
            'models': None
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
        log_info(f"Training Path: {str(training_path)}")

        return training_path

    def get_all_training_path(self) -> Dict:
        # >>>> TRAINING PATH
        self.training_path['ROOT'] = self.get_training_path(
            self.project_path, self.name)
        # >>>> MODEL PATH
        self.training_path['models'] = self.training_path['ROOT'] / 'models'
        # PARTITIONED DATASET PATH
        self.training_path['dataset'] = self.training_path['ROOT'] / 'dataset'
        # EXPORTED MODEL PATH
        self.training_path['exported_models'] = self.training_path['ROOT'] / \
            'exported_models'
        # ANNOTATIONS PATH
        self.training_path['annotations'] = self.training_path['ROOT'] / 'annotations'

        return self.training_path

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
            log_info(
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
                log_info(f"Removed Training Dataset {dataset}")

            except Exception as e:
                log_error(f"{e}")

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

            log_info(f"Inserted Training Dataset {dataset}")

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
                , partition_ratio = %s
            WHERE
                id = %s
            RETURNING
                id;
        
                                """
        partition_size_json = json.dumps(self.partition_ratio, indent=4)
        update_training_info_vars = [self.name,
                                     self.desc, partition_size_json, self.id]

        query_return = db_fetchone(update_training_info_SQL,
                                   conn,
                                   update_training_info_vars).id

        try:
            assert self.id == query_return, f'Updated wrong Training of ID {query_return}, which should be {self.id}'
            log_info(
                f"Successfully updated New Training Name, Desc and Partition Size for {self.id} ")
            return True

        except Exception as e:
            log_error(
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
                        , partition_ratio = %s
                    RETURNING
                        id;
                                    """

        partition_size_json = json.dumps(self.partition_ratio, indent=4)
        insert_training_info_vars = [
            self.name, self.desc, partition_size_json, self.project_id, self.desc, partition_size_json, ]

        try:
            query_return = db_fetchone(insert_training_info_SQL,
                                       conn,
                                       insert_training_info_vars).id
            self.id = query_return
            log_info(
                f"Successfully load New Training Name, Desc and Partition Size for {self.id} ")
            return True

        except TypeError as e:
            log_error(
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
            log_error(f"{e}")
            traceback.print_exc()

            return False


class NewTraining(BaseTraining):
    def __init__(self, training_id, project: Project) -> None:
        super().__init__(training_id, project)
        self.has_submitted: Dict = {
            NewTrainingPagination.InfoDataset: False,
            NewTrainingPagination.Model: False,
            NewTrainingPagination.TrainingConfig: False,
            NewTrainingPagination.AugmentationConfig: False
        }

        self.model_selected: NewModel = None  # DEPRECATED
        self.attached_model: Model = None
        self.training_model: NewModel = None
    # TODO *************************************

    # Wrapper for check_if_exists function from form_manager.py
    def check_if_exists(self, context: Dict, conn) -> bool:
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

        # >>>> TRAINING PATH
        self.training_path['ROOT'] = self.get_training_path(
            self.project_path, self.name)
        # >>>> MODEL PATH
        self.training_path['models'] = self.training_path['ROOT'] / 'models'
        # PARTITIONED DATASET PATH
        self.training_path['dataset'] = self.training_path['ROOT'] / 'dataset'
        # EXPORTED MODEL PATH
        self.training_path['exported_models'] = self.training_path['ROOT'] / \
            'exported_models'
        # ANNOTATIONS PATH
        self.training_path['annotations'] = self.training_path['ROOT'] / 'annotations'

        # >>>> CREATE Training directory recursively
        for path in self.training_path.values():
            create_folder_if_not_exist(path)
            log_info(
                f"Successfully created **{str(path)}**")

        # >>>> ASSERT IF TRAINING DIRECTORY CREATED CORRECTLY
        try:
            assert (len([x for x in self.training_path['ROOT'].iterdir() if x.is_dir()]) == (
                len(self.training_path) - 1)), f"Failed to create all folders"
            log_info(
                f"Successfully created **{self.name}** training at {str(self.training_path['ROOT'])}")

        except AssertionError as e:

            # Create a list of sub-folders excluding Training Root
            directory_structure = [
                x for x in self.training_path.values() if x != self.training_path['ROOT']]

            missing_folders = [x for x in self.training_path['ROOT'].iterdir(
            ) if x.is_dir() and (x not in directory_structure)]

            log_error(f"{e}: Missing {missing_folders}")

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

            # Create Training Folder
            self.initialise_training_folder()

            return True

        except AssertionError as e:
            log_error(f'{e}')
            traceback.print_exc()
            st.error(f'{e}')
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
            log_error(f"At update training_attached: {e}")
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

        new_training_attributes = ["new_training", "new_training_name",
                                   "new_training_desc", "new_training_model_page", "new_training_model_chosen"]

        reset_page_attributes(new_training_attributes)


class Training(BaseTraining):

    def __init__(self, training_id, project: Project) -> None:
        super().__init__(training_id, project)

        self.query_all_fields()
        self.training_path = self.get_all_training_path()
        self.attached_model: Model = None
        self.project_model: Model = None

        # TODO #136 query training details
        # get model attached
        # is_started
        # progress

    def query_all_fields(self) -> NamedTuple:
        """Query fields of current Training
        - name
        - description
        - training_param
        - augmentation
        - is_started
        - progress
        - partition_ratio

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
                self.training_param_dict, self.augmentation_dict,\
                self.is_started, self.progress, self.partition_ratio = training_field
        else:
            log_error(
                f"Training with ID {self.id} for Project ID {self.project_id} does not exists in the Database!!!")

            training_field = None

        return training_field

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
        log_info(
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

            log_error(f"""{e}: Could not find any training for Project ID {project_id} / \
                            Project ID: Project ID {project_id} does not exists""")

        return all_project_training, column_names

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
    #         log_info(
    #             f"Successfully created **{str(dir)}**")

    #     log_info(
    #         f"Successfully created **{self.name}** training at {str(self.training_path)}")

    #     # Entry into DB for training,project_training,model,model_training tables
    #     if self.insert_training():

    #         log_info(
    #             f"Successfully stored **{self.name}** traiing information in database")
    #         return True

    #     else:
    #         log_error(
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
        log_info(f"Formatting Date/Time and Progress column......")

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

    @staticmethod
    def reset_training_page():
        training_attributes = ["project_training_table", "training", "training_name",
                               "training_desc", "labelling_pagination", "existing_training_pagination"]

        reset_page_attributes(training_attributes)


def main():
    print("Hi")


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
