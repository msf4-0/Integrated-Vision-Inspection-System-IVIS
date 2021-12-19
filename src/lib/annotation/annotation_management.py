"""
Title: Annotation Manager
Date: 15/7/2021
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
import os
import shutil
import sys
from base64 import b64encode
from collections import namedtuple
from datetime import datetime
from enum import IntEnum
from io import BytesIO
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import numpy as np
import pandas as pd
import psycopg2
import streamlit as st
from PIL import Image
from psycopg2.extras import Json
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

from core.utils.dataset_handler import data_url_encoder_cv2, load_image_PIL
from core.utils.log import logger  # logger
from core.utils.form_manager import reset_page_attributes
from data_manager.database_manager import (db_fetchall, db_fetchone,
                                           db_no_fetch, init_connection)
from data_manager.dataset_management import (Dataset, FileTypes,
                                             data_url_encoder, load_image)
from core.utils.helper import create_dataframe, dataframe2dict, get_dataframe_row

# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from user.user_management import User

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<
conn = init_connection(**st.secrets['postgres'])


class AnnotationType(IntEnum):
    Image_Classification = 1
    BBox = 2
    Polygons = 3
    Masks = 4

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return AnnotationType[s]
        except KeyError:
            raise ValueError()


# Dictionary to store labels into DB
annotation_types = {
    AnnotationType.Image_Classification: 'Classification',
    AnnotationType.BBox: 'Bounding Box',
    AnnotationType.Polygons: 'Segmentation Polygon',
    AnnotationType.Masks: 'Segmentation Mask'
}


class LabellingPagination(IntEnum):
    AllTask = 0
    Labelled = 1
    Queue = 2
    Editor = 3
    Performance = 4
    EditorConfig = 5
    Export = 6

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return LabellingPagination[s]
        except KeyError:
            raise ValueError()


class BaseTask:
    def __init__(self) -> None:
        self.id: int = None
        self.data: Union[Dict[str], List[Dict]] = None  # Path to image
        self.meta: Dict = None
        self.project_id: int = None
        self.created_at: datetime = datetime.now().astimezone()
        self.updated_at: datetime = datetime.now().astimezone()
        self.is_labelled: bool = None
        self.overlap: int = None  # number of overlaps
        # self.file_upload:str=None NOTE: replaced with 'name'
        self.annotations: List[Dict] = None
        self.predictions: List[Dict] = None
        self.skipped: bool = False

        # extra
        self.dataset_id: int = None
        self.name: str = None
        self.data_list: Dict = {}


class NewTask(BaseTask):

    # create new Task
    @staticmethod
    def insert_new_task(image_name: str, project_id: int, dataset_id: int) -> int:
        insert_new_task_SQL = """
                                INSERT INTO public.task (
                                    name,
                                    project_id,
                                    dataset_id)
                                VALUES (
                                    %s,
                                    %s,
                                    %s)
                                ON CONFLICT ON CONSTRAINT task_pkey DO NOTHING

                                RETURNING id;
                                        """
        insert_new_task_vars = [image_name, project_id, dataset_id]
        task_id = db_fetchone(insert_new_task_SQL, conn,
                              insert_new_task_vars).id

        return task_id


class Task(BaseTask):
    def __init__(self, task_row: dict, dataset_dict: Dict[namedtuple, Any], project_id: int, annotations=None, predictions=None,
                 generate_data_url: bool = True) -> None:
        super().__init__()
        """
        - task_row: id,Task Name, Created By, Dataset Name, Is Labelled, Skipped, Date/Time
        - dataset_dict (from project class object)-> namedtuple: 
            - ID, Name,Dataset_Size,File_Type,Date_Time
        
        """
        # self.task_row = task_row
        self.dataset_dict = dataset_dict
        self.project_id: int = project_id
        self.annotations: List = annotations
        self.predictions: List = predictions

        # generate when call LS format generator
        self.data: dict = {'image': None}
        self.id = task_row['id']
        self.name: str = task_row['Task Name']
        self.dataset_name: str = task_row['Dataset Name']
        self.is_labelled: bool = task_row['Is Labelled']
        self.skipped: bool = task_row["Skipped"]

        self.dataset_id: int = dataset_dict[self.dataset_name].ID
        self.dataset_path: Path = Dataset.get_dataset_path(self.dataset_name)

        self.data_path: Path = self.dataset_path / self.name
        # self.filetype = Dataset.get_filetype_enumerator(self.name)
        self.filetype: FileTypes = FileTypes.Image

        if generate_data_url:
            # only need this to show the image on Label Studio Editor
            self.data_url: str = self.get_data()  # Get Data URL

        # self.query_task()

    @staticmethod
    def check_if_task_exists(image_name: str, project_id: int, dataset_id: int, conn=conn) -> bool:
        """Check if tasks exist in the Task table


        Args:
            image_name (str): Name of data
            project_id (int): Project ID
            dataset_id (int): Dataset ID
            conn (psycopg2.connection): Connection class object. Defaults to conn.

        Returns:
            bool: True if exists else False
        """
        check_if_exists_SQL = """
                                SELECT
                                    EXISTS (
                                        SELECT
                                            *
                                        FROM
                                            public.task
                                        WHERE
                                            name = %s
                                            AND project_id = %s
                                            AND dataset_id = %s);"""
        check_if_exists_vars = [image_name, project_id, dataset_id]
        exists_flag = db_fetchone(
            check_if_exists_SQL, conn, check_if_exists_vars).exists

        return exists_flag

    def query_task(self):
        """Query ID, 'Is Labelled' flag and 'Skipped' flag

        Returns:
            [type]: [description]
        """
        query_task_SQL = """
                        SELECT
                            id,
                            is_labelled,
                            skipped,
                            created_at,
                            updated_at
                        FROM
                            public.task
                        WHERE
                            name = %s
                            AND project_id = %s
                            AND dataset_id = %s;
                                """
        query_task_vars = [self.name, self.project_id, self.dataset_id]
        query_return = db_fetchone(
            query_task_SQL, conn, query_task_vars, fetch_col_name=False)
        try:
            self.id, self.is_labelled, self.skipped, self.created_at, self.updated_at = query_return
            return query_return
        except TypeError as e:
            logger.error(
                f"{e}: Task for data {self.name} from Dataset {self.dataset_id} does not exist in table for Project {self.project_id}")

    @staticmethod
    def update_task_dict(all_task: List[Dict[str, Any]], task_id: int,
                         labelled: bool = False, skipped: bool = False,
                         for_data_table: bool = True):
        ID_string = "id" if for_data_table else "ID"
        for task in all_task:
            if task[ID_string] == task_id:
                task["Is Labelled"] = labelled
                task["Skipped"] = skipped
                break
        else:
            logger.error(f"'task_id' {task_id} does not exist in 'all_task'")

    def generate_data_url(self) -> str:
        """Generate data url from OpenCV numpy array

        Returns:
            str: Data URl with base64 bytes encoded in UTF-8
        """

        try:
            self.data_url = data_url_encoder(
                # self.data_object, self.filetype, self.data_path)
                self.filetype, self.data_path)

            return self.data_url
        except Exception as e:
            logger.error(f"{e}: Failed to generate data url for {self.name}")
            return None

    # @st.cache
    def load_data(self):
        if self.filetype == FileTypes.Image:
            self.data_object = load_image(self.data_path, opencv_flag=True)

        elif self.filetype == FileTypes.Video:
            pass
        elif self.filetype == FileTypes.Audio:
            pass
        elif self.filetype == FileTypes.Text:
            pass

        return self.data_object

    # @st.cache
    def get_data(self):

        # data = self.data_list.get(self.name)

        # if not data:
        # self.data_object = self.load_data()
        self.data_url = self.generate_data_url()
        # add encoded image into dictionary
        # self.data_list[self.name] = data

        # else:
        # logger.info(f"Data {self.name} EXIST")

        return self.data_url

    def generate_editor_format(self, annotations_dict: List, predictions_dict: List = None):
        """Generate editor format based on Label Studio Frontend requirements

        Args:
            annotations_dict ([type]): [description]
            predictions_dict ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        try:
            # task = {
            #     'id': self.id,
            #     'annotations': [annotations_dict],
            #     'predictions': [predictions_dict],
            #     'file_upload': self.name,
            #     # "data": {"image": self.generate_data_url()},
            #     'data': {'image': None},
            #     'meta': {},
            #     'created_at': str(self.created_at),
            #     'updated_at': str(self.updated_at),
            #     'project': self.project_id
            # }
            # data = self.get_data()

            task = {
                'id': self.id,
                'annotations': annotations_dict,
                'predictions': [],
                # 'file_upload': self.name,
                "data": {"image": self.data_url},  # NOTE
                # 'data': {'image': None}
                # 'meta': {},
                # 'created_at': str(self.created_at),
                # 'updated_at': str(self.updated_at),
                # 'project': self.project_id
            }
            return task
        except Exception as e:
            logger.error(f"{e}: Failed to generate editor format for editor")
            task = {}
            return task

    @staticmethod
    def delete_task(taskname: str):
        delete_task_SQL = """
                            DELETE FROM public.task
                            WHERE task.name = %s
                            RETURNING
                                *;
                                """
        delete_task_vars = [taskname]
        query_delete_return = db_fetchone(
            delete_task_SQL, conn, delete_task_vars)

        # return to confirm task is deleted
        return query_delete_return

    @staticmethod
    def query_all_task(project_id: int, return_dict: bool = False, for_data_table: bool = False) -> Union[List[namedtuple], List[dict]]:

        ID_string = "id" if for_data_table else "ID"
        query_all_task_SQL = f"""
                    SELECT
                        t.id AS \"{ID_string}\",
                        t.name AS "Task Name",
                        CASE WHEN a.users_id IS NULL THEN
                            '-'
                        ELSE
                            u.first_name || ' ' || u.last_name
                        END AS "Created By",
                        (
                            SELECT
                                d.name AS "Dataset Name"
                            FROM
                                public.dataset d
                            WHERE
                                d.id = t.dataset_id),
                        t.is_labelled AS "Is Labelled",
                        t.skipped AS "Skipped",
                        to_char(t.updated_at, 'YYYY-MM-DD HH24:MI:SS')  AS "Date/Time"
                    FROM
                        public.task t
                        LEFT JOIN public.annotations a ON a.id = t.annotation_id
                        LEFT JOIN public.users u ON u.id = a.users_id
                    WHERE
                        t.project_id = %s
                    ORDER BY
                        t.id;
                                """
        query_all_task_vars = [project_id]
        logger.info(
            f"Querying annotations and tasks from database for Project {project_id}")

        all_task_query_return, column_names = db_fetchall(
            query_all_task_SQL, conn, query_all_task_vars,
            fetch_col_name=True,
            return_dict=return_dict)
        return all_task_query_return, column_names

    @staticmethod
    def create_all_task_dataframe(all_task: Union[List[namedtuple], List[dict]], column_names: List = None) -> pd.DataFrame:
        """Generate Pandas DataFrame to store Annotation Task query

        Args:
            all_task (Union[List[namedtuple], List[dict]]): Annotation Task query from 'query_all_task()'
            column_names (List, optional): Names of columns. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame of Annotation Task query
        """
        # create DF for All Task Table
        # returns Pandas DataFrame
        df = create_dataframe(all_task, column_names, date_time_format=True)
        df['Created By'] = df['Created By'].fillna("-")
        df['Date/Time'] = df['Date/Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        return df

    @staticmethod
    @dataframe2dict(orient='index')
    def get_labelled_task(all_task: Union[List[namedtuple], List[dict]],
                          is_labelled: bool = True) -> List[Dict]:
        """Get a List of Task-Annotations Dict where is_labelled is True

        Args:
            all_task (Union[List[namedtuple], List[dict]]): all_task query from 'get_all_task()'
            is_labelled (bool, optional): Labelled data means either 'Is Labelled' or 'Skipped. Defaults to True.
            is_skipped_only (bool, optional): If True, only return skipped data. Defaults to False

        Returns:
            List[Dict]: List of dictionaries based on Material UI Data Grid format
        """

        all_task_df = Task.create_all_task_dataframe(all_task)
        if is_labelled:
            filt = ((all_task_df['Is Labelled'] == True)
                    | (all_task_df['Skipped'] == True))
        else:
            filt = ((all_task_df['Is Labelled'] == False)
                    & (all_task_df['Skipped'] == False))
        labelled_task_df = all_task_df.loc[filt]
        # labelled_task_df=labelled_task_df[["id","Task Name","Created By","Dataset Name","Date/Time"]]

        # labelled_task_dict = list(
        #     (labelled_task_df.to_dict(orient='index')).values())

        return labelled_task_df

    @staticmethod
    def query_next_task(project_id: int, return_dict: bool = True,
                        for_data_table: bool = True
                        ) -> Union[Dict[str, Any], NamedTuple]:
        """
        Get the next unlabeled task row (not labeled and not skipped) for the `project_id`
        to help the user to proceed to next task automatically, and MUCH more faster.

        Returns `None` if there is no more unlabeled task.
        """
        # this query must follow the data from self.query_all_task
        ID_string = "id" if for_data_table else "ID"
        sql_query = f"""
            SELECT
                t.id AS \"{ID_string}\",
                t.name AS "Task Name",
                CASE WHEN a.users_id IS NULL THEN
                    '-'
                ELSE
                    u.first_name || ' ' || u.last_name
                END AS "Created By",
                (
                    SELECT
                        d.name AS "Dataset Name"
                    FROM
                        public.dataset d
                    WHERE
                        d.id = t.dataset_id),
                t.is_labelled AS "Is Labelled",
                t.skipped AS "Skipped",
                to_char(t.updated_at, 'YYYY-MM-DD HH24:MI:SS')  AS "Date/Time"
            FROM
                public.task t
                LEFT JOIN public.annotations a ON a.id = t.annotation_id
                LEFT JOIN public.users u ON u.id = a.users_id
            WHERE
                t.project_id = %s
                and (is_labelled = False and skipped = False)
            ORDER BY
                t.id
            LIMIT 1;
        """
        query_vars = [project_id]
        record = db_fetchone(sql_query, conn, query_vars,
                             return_dict=return_dict)  # return tuple
        if not record:
            logger.debug(f"No more unlabeled task for project ID "
                         f"{project_id} :)")
        return record


class Result:
    def __init__(self, from_name, to_name, type, value) -> None:
        self.from_name: str = from_name
        self.to_name: str = to_name
        self.type: str = type
        self.value: List[Dict] = value


class BasePredictions:
    # will then be converted into JSON format
    def __init__(self) -> None:
        self.id: int = None  # predictions_id
        self.model_version: str = None
        self.created_ago: datetime = None
        self.completed_by: Dict = {}  # user_id, email, first_name, last_name
        self.was_cancelled: bool = False
        self.ground_truth: bool = True
        self.result: List[Dict, Result] = []  # or Dict?
        self.score: float = None


class NewPredictions(BasePredictions):
    def __init__(self) -> None:
        super().__init__()


class Predictions(BasePredictions):
    def __init__(self) -> None:
        super().__init__()


class BaseAnnotations:
    # will then be converted into JSON format
    def __init__(self, task: Task) -> None:
        self.id: int = None  # annotations_id
        # user_id, email, first_name, last_name
        self.completed_by: Dict = {
            "id": None, "email": None, "first_name": None, "last_name": None}
        self.was_cancelled: bool = False
        self.ground_truth: bool = False
        self.created_at: datetime = datetime.now().astimezone()
        self.updated_at: datetime = datetime.now().astimezone()
        self.lead_time: float = 0
        self.task: Task = task  # equivalent to 'task_id'
        self.result: List[Dict, Result] = []  # or Dict?
        self.predictions: List[Dict, Predictions] = None

    def submit_annotations(self, result: Dict, users_id: int, conn=conn) -> int:
        """ Submit result for new annotations

        Args:
            result (Dict): [description]
            project_id (int): [description]
            users_id (int): [description]
            task_id (int): [description]
            is_labelled (bool, optional): [description]. Defaults to True.
            conn (psycopg2 connection object, optional): [description]. Defaults to conn.

        Returns:
            [type]: [description]
        """
        logger.debug(f"Submitting results.......")
        # NOTE: Update class object: result + task
        self.result = result if result else None
        self.task.is_labelled = True
        # TODO is it neccessary to have annotation type id?
        insert_annotations_SQL = """
                                    INSERT INTO public.annotations (
                                        result,
                                        project_id,
                                        users_id,
                                        task_id)
                                    VALUES (
                                        %s::JSONB[],
                                        %s,
                                        %s,
                                        %s)
                                    RETURNING id,result;
                                """
        # insert_annotations_vars = [json.dumps(
        #     result), self.task.project_id, users_id, self.task.id]
        result_serialised = [json.dumps(x) for x in result]
        insert_annotations_vars = [result_serialised,
                                   self.task.project_id, users_id, self.task.id]
        try:
            self.id, self.result = db_fetchone(
                insert_annotations_SQL, conn, insert_annotations_vars)
        except psycopg2.Error as e:
            error = e.pgcode
            logger.error(f"{error}: Annotations already exist")

        # NOTE: Update 'task' table with annotation id and set is_labelled flag as True
        update_task_SQL = """
                            UPDATE
                                public.task
                            SET
                                annotation_id = %s,
                                is_labelled = True,
                                skipped = False
                            WHERE
                                id = %s
                            RETURNING is_labelled;
                        """

        update_task_vars = [self.id, self.task.id]

        try:
            self.task.is_labelled = db_fetchone(
                update_task_SQL, conn, update_task_vars)
        except TypeError as e:
            logger.error(f"{e}: Task update for Submit failed")

        # sleep(1)
        return self.result

    def update_annotations(self, result: Dict, users_id: int, conn=conn) -> tuple:
        """Update result for new annotations

        Args:
            result (Dict): [description]
            users_id (int): [description]
            conn (psycopg2 connection object, optional): [description]. Defaults to conn.

        Returns:
            tuple: [description]
        """

        logger.info(f"Updating results")
        self.result = result if result else None  # update result attribute
        result_serialised = [json.dumps(x) for x in result]

        update_annotations_SQL = """
                                    UPDATE
                                        public.annotations
                                    SET
                                        result = %s::JSONB[],
                                        users_id = %s
                                    WHERE
                                        id = %s
                                    RETURNING id,result;
                                """

        update_annotations_vars = [result_serialised, users_id, self.id]
        # updated_annotation_return = db_fetchone(
        #     update_annotations_SQL, conn, update_annotations_vars)
        try:
            updated_annotation_return = db_fetchone(
                update_annotations_SQL, conn, update_annotations_vars)
            self.id, self.result = updated_annotation_return

# NEW************************************
            logger.info(
                f"Update annotations for Task {self.task.name} with Annotation ID: {self.id}")
            return updated_annotation_return
        except psycopg2.Error as e:
            error = e.pgcode
            logger.error(f"{error}: Annotations already exist")

    def delete_annotation(self, conn=conn) -> tuple:
        """Delete annotations

        Returns:
            tuple: [description]
        """
        delete_annotations_SQL = """
                                DELETE FROM public.annotations
                                WHERE id = %s
                                RETURNING *;
                                """
        delete_annotations_vars = [self.id]

        delete_annotation_return = db_fetchone(
            delete_annotations_SQL, conn, delete_annotations_vars)
        self.result = []
        return delete_annotation_return

    def skip_task(self, skipped: bool = True, conn=conn) -> tuple:
        """Skip task

        Args:
            task_id (int): [description]
            skipped (bool): [description]

        Returns:
            tuple: [description]
        """
        skip_task_SQL = """
                        UPDATE
                            public.task
                        SET
                            skipped = %s,
                            is_labelled = False
                        WHERE
                            id = %s
                        RETURNING *;
                    """
        skip_task_vars = [
            skipped, self.task.id]  # should set 'skipped' as True
        logger.info(self.task.id)
        skipped_task_return = db_fetchone(skip_task_SQL, conn, skip_task_vars)

        # must delete the previously created annotation if any exists
        delete_annotation_return = self.delete_annotation()

        return delete_annotation_return

    def generate_annotation_dict(self) -> Union[Dict, List]:
        try:
            if self.result:
                annotation_dict = [{"id": self.id,
                                    "completed_by": self.completed_by,
                                    "result": self.result,
                                    "was_cancelled": self.was_cancelled,
                                    "ground_truth": self.ground_truth,
                                    "created_at": str(self.created_at),
                                    "updated_at": str(self.updated_at),
                                    "lead_time": str(self.updated_at - self.created_at),
                                    "prediction": {},
                                    "result_count": 0,
                                    "task": self.task.id
                                    }]
            else:
                annotation_dict = []
            return annotation_dict
        except Exception as e:
            logger.error(
                f"{e}: Failed to generate annotation dict for Annotation {self.id}")
            annotation_dict = []
            return annotation_dict


class NewAnnotations(BaseAnnotations):
    def __init__(self, task: Task, user: User) -> None:
        super().__init__(task)
        self.task: Task = task
        # Current user
        self.user: User = user

    # NOTE REDUNDANT
    # def generate_annotation_dict(self) -> Union[Dict, List]:
    #     try:
    #         annotation_dict = []
    #         return annotation_dict
    #     except Exception as e:
    #         logger.error(
    #             f"{e}: Failed to generate annotation dict for Annotation {self.id}")
    #         annotation_dict = []
    #         return annotation_dict


class Annotations(BaseAnnotations):
    def __init__(self, task: Task) -> None:
        super().__init__(task)

        logger.info(f"Initialising Existing Annotations {self.task.name}")

        self.task = task  # 'Task' class object
        self.user = {}
        if self.task.is_labelled:
            self.query_annotations()  # get annotations id, results, user details, date/time
            self.completed_by = {
                "id": self.user["id"],
                "email": self.user["email"],
                "first_name": self.user["first_name"],
                "last_name": self.user["last_name"]
            }
        else:
            pass

    @staticmethod
    def check_if_annotation_exists(task_id: int, project_id: int, conn=conn) -> bool:
        """Check if annotation exists in annotations table

        Args:
            task_id (int): Task ID
            project_id (int): Project ID
            conn (psycopg2.connection, optional): PostgreSQL psycopg2 connection object. Defaults to conn.

        Returns:
            bool: True if annotation exist, False otherwise
        """
        check_if_exists_SQL = """
                                SELECT
                                    EXISTS (
                                        SELECT
                                            *
                                        FROM
                                            public.annotations
                                        WHERE
                                            task_id = %s
                                            AND project_id = %s
                                            );"""
        check_if_exists_vars = [task_id, project_id]
        exists_flag = db_fetchone(
            check_if_exists_SQL, conn, check_if_exists_vars).exists

        return exists_flag

    def query_annotations(self) -> namedtuple:
        """Query all columns annotations table

        Returns:
            namedtuple: Query return for annotations
        """

        query_annotation_SQL = """
                                SELECT
                                    a.id,
                                    result,
                                    u.id as user_id,u.email,u.first_name,u.last_name,
                                    a.created_at,
                                    a.updated_at
                                FROM
                                    public.annotations a
                                INNER JOIN public.users u
                                ON a.users_id = u.id
                                WHERE
                                    task_id = %s;

                                """
        query_annotation_vars = [self.task.id]
        query_return = db_fetchone(
            query_annotation_SQL, conn, query_annotation_vars, fetch_col_name=False)

        try:

            self.id, self.result, self.user["id"], self.user["email"], self.user[
                "first_name"], self.user["last_name"], self.created_at, self.updated_at = query_return
            logger.info("Query annotations in class")

            return query_return

        except TypeError as e:
            logger.error(
                f"{e}: Annotation for Task {self.task.id} from Dataset {self.task.dataset_id} does not exist in table for Project {self.task.project_id}")


def get_task_row(task_id: int, task_df: pd.DataFrame) -> Dict:
    """Get data name from task_df dataframe (id,Task Name, Is Labelled, Skipped)

    #### Disclaimer: Task name and Data name is interusable 

    Args:
        task_id (int): Task ID obtained from Data Table row selection
        task_df (pd.DataFrame): DataFrame from create_all_task_dataframe() method from Dataset class

    Raises:
        TypeError: If data_id is not type (int)

    Returns:
        dict: Row of data
    """
    # to obtain name data based on selection of data table
    logger.info(f"Obtaining data row from task_df")

    task_row = get_dataframe_row(task_id, task_df)

    logger.info(f"Currently serving data:{task_row['Task Name']}")

    return task_row


def reset_editor_page():
    """
    This function is used every time the user clicked on a navigation in
    the labelling_dashboard pages.

    So some states such as "labelling_pagination", and "all_task" are not included 
    here so that we can keep them across pages. They are included in 
    Training.reset_training_page()
    """
    if session_state.get('zipfile_path'):
        if os.path.exists(session_state['zipfile_path']):
            logger.debug("Removing exported zipfile: "
                         f"{session_state['zipfile_path']}")
            # don't need it anymore
            os.remove(session_state['zipfile_path'])
    if 'project' in session_state:
        # must clear this dataset path just in case the user changes any annotations
        dataset_export_path = session_state.project.get_export_path()
        if dataset_export_path.exists():
            logger.debug(
                f"Removing dataset export path: {dataset_export_path}")
            shutil.rmtree(dataset_export_path)

    editor_attributes = ["new_annotation_flag", "task", "show_next_unlabeled"
                         "annotation", "data_labelling_table", "labelling_prev_result"
                         'data_selection', 'zipfile_path']

    logger.debug(f"Resetting Editor Page......")
    reset_page_attributes(editor_attributes)


# ******************************** DATA TABLE COLUMNS CONFIG ********************************************
task_labelling_columns = [
    {
        'field': "id",
        'headerName': "ID",
        'headerAlign': "center",
        'align': "center",
        'flex': 70,
        'hideSortIcons': True,

    },
    {
        'field': "Task Name",
        'headerAlign': "center",
        'align': "center",
        'flex': 200,
        'hideSortIcons': False,
    },

    {
        'field': "Dataset Name",
        'headerAlign': "center",
        'align': "center",
        'flex': 150,
        'hideSortIcons': False,
    },
    {
        'field': "Is Labelled",
        'headerAlign': "center",
        'align': "center",
        'flex': 100,
        'hideSortIcons': True,
        'type': 'boolean',
    },
    {
        'field': "Skipped",
        'headerAlign': "center",
        'align': "center",
        'flex': 100,
        'hideSortIcons': True,
        'type': 'boolean',
    },


]

# *********************************NOTE REDUNDANT *********************************


@st.cache
def load_buffer_image():
    """Load Image and generate Data URL in base64 bytes

    Args:
        image (bytes-like): BytesIO object

    Returns:
        bytes: UTF-8 encoded base64 bytes
    """
    chdir_root()  # ./image_labelling
    logger.info("Loading Sample Image")
    sample_image = "resources/buffer.png"
    with Image.open(sample_image) as img:
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format=img.format)

    bb = img_byte_arr.getvalue()
    b64code = b64encode(bb).decode('utf-8')
    data_url = 'data:image/jpeg;base64,' + b64code
    # data_url = f'data:image/jpeg;base64,{b64code}'
    # st.write(f"\"{data_url}\"")

    return data_url


# NOTE REDUNDANT*********************************************************************************
def get_data_name(data_id: int, task_df: pd.DataFrame) -> str:
    """Get data name from task_df dataframe (id,Task Name, Is Labelled, Skipped)

    #### Disclaimer: Task name and Data name is interusable 

    Args:
        data_id (int): Task ID obtained from Data Table row selection
        task_df (pd.DataFrame): DataFrame from create_all_task_dataframe() method from Dataset class

    Raises:
        TypeError: If data_id is not type (int)

    Returns:
        str: Name of data
    """
    # to obtain name data based on selection of data table
    logger.info(f"Obtaining data name from task_df")

    # Handle data_id exceptions
    if isinstance(data_id, int):
        pass
    elif isinstance(data_id, Union[List, tuple]):
        assert len(data_id) <= 1, "Data selection should be singular"
        data_id = data_id[0]
    else:
        raise TypeError("Data ID can only be int")

    data_name = (task_df.loc[task_df["id"] == data_id])["Task Name"].values[0]

    logger.info(f"Currently serving data:{data_name}")

    return data_name


# NOTE DEPRECATED *****************************************************************************
@st.cache
def get_first_task_row(task_df: pd.DataFrame):
    # For loading of interface from 'Start Labelling' where no data is selected
    # Load first element
    first_data_row = task_df.iloc[0].to_dict()

    return first_data_row

# NOTE DEPRECATED *****************************************************************************


def get_image_size(image_path):
    """get dimension of image

    Args:
        image_path (str): path to image or byte_like object

    Returns:
        tuple: original_width and original_height
    """
    with Image.open(image_path) as img:
        original_width, original_height = img.size
    return original_width, original_height
