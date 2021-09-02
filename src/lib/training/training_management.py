"""
Title: Training Management
Date: 23/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import json
import sys
from enum import IntEnum
from pathlib import Path
from time import sleep
from typing import Dict, List, Optional, Union

import pandas as pd
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as SessionState

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

from core.utils.file_handler import create_folder_if_not_exist
from core.utils.form_manager import (check_if_exists, check_if_field_empty,
                                     reset_page_attributes)
from core.utils.helper import datetime_formatter, get_directory_name
from core.utils.log import log_error, log_info  # logger
from data_manager.database_manager import (db_fetchall, db_fetchone,
                                           db_no_fetch, init_connection)
from project.project_management import Project
from training.model_management import Model, NewModel

# >>>> User-defined Modules >>>>


# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration >>>>

# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


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
        self.model_id: int = None
        self.pre_trained_model_id: Optional[int] = None
        self.deployment_type: int = project.deployment_type
        self.model = None  # TODO
        self.model_path: str = None
        self.framework: str = None
        self.partition_ratio: float = 0.5
        self.dataset_chosen: List = None
        self.training_param_json: Dict = None
        self.augmentation_json: Dict = None
        self.is_started: bool = False
        self.progress: Dict = {}

    @st.cache
    def get_framework_list(self):
        get_framework_list_SQL = """
            SELECT
                id as "ID",
                name as "Name"
            FROM
                public.framework;
                    """
        framework_list = db_fetchall(get_framework_list_SQL, conn)
        return framework_list


class NewTraining(BaseTraining):
    def __init__(self, training_id, project: Project) -> None:
        super().__init__(training_id, project)
        self.has_submitted = False
        self.model_selected = None  # TODO
    # TODO *************************************

# TODO #109 Update Check if exist and check if field exist

    # Wrapper for check_if_exists function from form_manager.py
    def check_if_exists(self, context: Dict, conn) -> bool:
        table = 'public.training'

        exists_flag = check_if_exists(
            table, context['column_name'], context['value'], conn)

        # return True if exists
        return exists_flag

    # Wrapper for check_if_exists function from form_manager.py
    def check_if_field_empty(self, context: Dict, field_placeholder):
        check_if_exists = self.check_if_exists
        empty_fields = check_if_field_empty(
            context, field_placeholder, check_if_exists)

        # True if not empty, False otherwise
        return empty_fields

    # NOTE DEPRECATED
    # TODO Remove
    def check_if_field_empty(self, field: List, field_placeholder) -> bool:
        empty_fields = []
        keys = ["name", "dataset_chosen", "model"]
        # if not all_field_filled:  # IF there are blank fields, iterate and produce error message
        for i in field:
            if i and i != "":
                if field.index(i) == 0:
                    context = ['name', field[0]]
                    if self.check_if_exist(context, conn):
                        field_placeholder[keys[0]].error(
                            f"Project name used. Please enter a new name")
                        log_error(
                            f"Project name used. Please enter a new name")
                        empty_fields.append(keys[0])

                else:
                    pass
            else:

                idx = field.index(i)
                field_placeholder[keys[idx]].error(
                    f"Please do not leave field blank")
                empty_fields.append(keys[idx])

        # if empty_fields not empty -> return False, else -> return True
        return not empty_fields

    # NOTE DEPRECATED
    # TODO Remove
    def check_if_exist(self, context: List, conn) -> bool:
        check_exist_SQL = """
                            SELECT
                                EXISTS (
                                    SELECT
                                        %s
                                    FROM
                                        public.training
                                    WHERE
                                        name = %s);
                        """
        exist_status = db_fetchone(check_exist_SQL, conn, context)[0]
        return exist_status

    def insert_training(self, model: Model, project: Project):

        # insert into training table
        insert_training_SQL = """
            INSERT INTO public.training (
                name,
                description,
                training_param,
                augmentation,
                pre_trained_model_id,
                framework_id,
                project_id,
                partition_size)
            VALUES (
                %s,
                %s,
                %s::jsonb,
                %s::jsonb,
                (
                    SELECT
                        pt.id
                    FROM
                        public.pre_trained_models pt
                    WHERE
                        pt.name = %s), (
                        SELECT
                            f.id
                        FROM
                            public.framework f
                        WHERE
                            f.name = %s), %s, %s)
            RETURNING
                id;

            """
        # convert dictionary into serialised JSON
        self.training_param_json = json.dumps(
            self.training_param, sort_keys=True, indent=4)
        self.augmentation_json = json.dumps(
            self.augmentation, sort_keys=True, indent=4)
        insert_training_vars = [self.name, self.desc, self.training_param_json, self.augmentation_json,
                                self.model_selected, self.framework, self.project_id, self.partition_ratio]
        self.training_id = db_fetchone(
            insert_training_SQL, conn, insert_training_vars).id

        # Insert into project_training table
        insert_project_training_SQL = """
            INSERT INTO public.project_training (
                project_id,
                training_id)
            VALUES (
                %s,
                %s);
        
            """
        insert_project_training_vars = [self.project_id, self.training_id]
        db_no_fetch(insert_project_training_SQL, conn,
                    insert_project_training_vars)

        # insert into model table
        insert_model_SQL = """
            INSERT INTO public.models (
                name,
                model_path,
                training_id,
                framework_id,
                deployment_id)
            VALUES (
                %s,
                %s,
                %s,
                (
                    SELECT
                        f.id
                    FROM
                        public.framework f
                    WHERE
                        f.name = %s), (
                        SELECT
                            dt.id
                        FROM
                            public.deployment_type dt
                        WHERE
                            dt.name = %s))
            RETURNING
                id;
            """
        insert_model_vars = [model.name, model.model_path,
                             self.training_id, self.framework, project.deployment_type]
        model.id = db_fetchone(insert_model_SQL, conn, insert_model_vars)

        # Insert into training_dataset table
        insert_training_dataset_SQL = """
            INSERT INTO public.training_dataset (
                training_id,
                dataset_id)
            VALUES (
                %s,
                (
                    SELECT
                        id
                    FROM
                        public.dataset d
                    WHERE
                        d.name = %s))"""
        for dataset in self.dataset_chosen:
            insert_training_dataset_vars = [self.training_id, dataset]
            db_no_fetch(insert_training_dataset_SQL, conn,
                        insert_training_dataset_vars)

        return self.id

    # TODO #133 Add New Training Reset
    @staticmethod
    def reset_new_training_page():

        new_training_attributes = ["new_training", "new_training_name",
                                   "new_training_desc", "new_training_model_page", "new_training_model_chosen"]

        reset_page_attributes(new_training_attributes)


class Training(BaseTraining):

    def __init__(self, training_id, project: Project) -> None:
        super().__init__(training_id, project)

        # query training details
        # get model attached
        # is_started
        # progress

    @staticmethod
    def query_all_project_training(project_id: int, return_dict: bool = False, for_data_table: bool = False) -> Union[List[namedtuple], List[dict]]:

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

            all_project_training = datetime_formatter(
                all_project_training_query_return, return_dict=return_dict)

        except TypeError as e:

            log_error(f"""{e}: Could not find any training for Project ID {project_id} / \
                            Project ID: Project ID {project_id} does not exists""")

        return all_project_training, column_names

    def initialise_training(self, model: Model, project: Project):
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
        directory_name = get_directory_name(self.name)

        self.training_path = project.project_path / \
            str(directory_name)  # use training name
        self.main_model_path = self.training_path / 'models'
        self.dataset_path = self.training_path / 'dataset'
        self.exported_model_path = self.training_path / 'exported_models'
        self.annotations_path = self.training_path / 'annotations'
        directory_pipeline = [self.annotations_path, self.exported_model_path, self.main_model_path,
                              self.dataset_path]

        # CREATE Training directory recursively
        for dir in directory_pipeline:
            create_folder_if_not_exist(dir)
            log_info(
                f"Successfully created **{str(dir)}**")

        log_info(
            f"Successfully created **{self.name}** training at {str(self.training_path)}")

        # Entry into DB for training,project_training,model,model_training tables
        if self.insert_training():

            log_info(
                f"Successfully stored **{self.name}** traiing information in database")
            return True

        else:
            log_error(
                f"Failed to stored **{self.name}** training information in database")
            return False

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
