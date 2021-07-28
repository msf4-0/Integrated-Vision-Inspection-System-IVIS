"""
Title: Annotation Manager
Date: 15/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
from typing import List, Dict, Union
from datetime import datetime
import psycopg2
import json
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

# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
from data_manager.database_manager import db_no_fetch, init_connection, db_fetchone
from core.utils.helper import check_if_exists
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<
conn = init_connection(**st.secrets['postgres'])


class BaseTask:
    def __init__(self) -> None:
        self.id: int = None
        self.data: Union[Dict[str], List[Dict]] = None
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


class NewTask(BaseTask):
    def __init__(self) -> None:
        super().__init__()


class Task(BaseTask):
    def __init__(self) -> None:
        super().__init__()

    # TODO: check if current image exists as a 'task' in DB
    @staticmethod
    def check_if_task_exists(table: str, column_name: str, condition, conn=conn):
        """SELECT
                EXISTS (
                    SELECT
                        *
                    FROM
                        {schema,tablename} --from 'table'
                    WHERE
                        {column_name} = %s); -- %s=condition
                """
        return check_if_exists(table, column_name, condition, conn).exists


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
        self.ground_truth: bool = True
        self.created_at: datetime = datetime.now().astimezone()
        self.updated_at: datetime = datetime.now().astimezone()
        self.lead_time: float = 0
        self.task: Task = task  # equivalent to 'task_id'
        self.result: List[Dict, Result] = []  # or Dict?
        self.predictions: List[Dict, Predictions] = None


class NewAnnotations(BaseAnnotations):
    def __init__(self, task: Task) -> None:
        super().__init__(task)

    def submit_annotations(self, result: Dict, project_id: int, users_id: int, conn=conn) -> int:
        """ Submit result for new annotations

        Args:
            result (Dict): [description]
            project_id (int): [description]
            users_id (int): [description]
            task_id (int): [description]
            annotation_id (int): [description]
            is_labelled (bool, optional): [description]. Defaults to True.
            conn (psycopg2 connection object, optional): [description]. Defaults to conn.

        Returns:
            [type]: [description]
        """
        # NOTE: Update class object: result + task
        self.result = result if result else None

        # TODO is it neccessary to have annotation type id?
        insert_annotations_SQL = """
                                    INSERT INTO public.annotations (
                                        result,
                                        project_id,
                                        users_id,
                                        task_id)
                                    VALUES (
                                        %s::jsonb,
                                        %s,
                                        %s,
                                        %s)
                                    RETURNING id;
                                """
        insert_annotations_vars = [json.dumps(
            result), project_id, users_id, self.task.id]
        self.annotation_id = db_fetchone(
            insert_annotations_SQL, conn, insert_annotations_vars, insert_annotations_vars).id

        # NOTE: Update 'task' table with annotation id and set is_labelled flag as True
        update_task_SQL = """
                            UPDATE
                                public.task
                            SET
                                (annotation_id = %s),
                                (is_labelled = %s)
                            WHERE
                                id = %s;
                        """
        update_task_vars = [self.annotation_id,
                            self.task.is_labelled, self.task.id]
        db_no_fetch(update_task_SQL, conn, update_task_vars)

        return self.annotation_id


class Annotations(BaseAnnotations):
    def __init__(self, task: Task) -> None:
        super().__init__(task)

    def update_annotations(self, result: Dict, users_id: int, conn=conn) -> tuple:
        """Update result for new annotations

        Args:
            result (Dict): [description]
            users_id (int): [description]
            annotation_id (int): [description]
            conn (psycopg2 connection object, optional): [description]. Defaults to conn.

        Returns:
            tuple: [description]
        """

        # TODO is it neccessary to have annotation type id?
        update_annotations_SQL = """
                                    UPDATE
                                        public.annotations
                                    SET
                                        (result = %s::jsonb),
                                        (users_id = %s)
                                    WHERE
                                        id = %s
                                    RETURNING *;
                                """
        update_annotations_vars = [json.dumps(
            result), users_id, self.id]
        updated_annotation_return = db_fetchone(
            update_annotations_SQL, conn, update_annotations_vars)

        return updated_annotation_return

    def delete_annotation(self) -> tuple:
        """Delete annotations

        Args:
            annotation_id (int): [description]

        Returns:
            tuple: [description]
        """
        delete_annotations_SQL = """
                                DELETE FROM public.annotation
                                WHERE id = %s
                                RETURNING *;
                                """
        delete_annotations_vars = [self.id]

        delete_annotation_return = db_fetchone(
            delete_annotations_SQL, conn, delete_annotations_vars)

        return delete_annotation_return

    def skip_task(self, skipped: bool = True) -> tuple:
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
                            (skipped = %s)
                        WHERE
                            id = %s;
                    """
        skip_task_vars = [
            skipped, self.task.id]  # should set 'skipped' as True

        skipped_task_return = db_fetchone(skip_task_SQL, conn, skip_task_vars)

        return skipped_task_return
