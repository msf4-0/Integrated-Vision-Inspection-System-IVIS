"""
Title: Editor Manager
Date: 22/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

from os import name
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

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<
conn = init_connection(**st.secrets['postgres'])

# Editor table
# - id
# - name
# - editor_config
# - labels
# - project_id


class BaseEditor:
    def __init__(self, random_generator) -> None:

        # name is editor id for reference. Not the same as PK of DB
        self.name: str = random_generator
        self.editor_config: str = None
        self.labels: List = []
        self.project_id: Union[str, int] = None


class NewEditor(BaseEditor):
    def __init__(self, random_generator) -> None:
        super().__init__(random_generator)

# ************************************************* OLD *************************************************


class Results:
    def __init__(self, from_name, to_name, type, value) -> None:
        self.from_name: str = from_name
        self.to_name: str = to_name
        self.type: str = type
        self.value: List[Dict] = value


class Annotations:
    def __init__(self) -> None:
        self.id: int = 0
        self.completed_by: Dict = {}  # user_id, email, first_name, last_name
        self.was_cancelled: bool = False
        self.ground_truth: bool = True
        self.created_at: datetime = datetime.now().astimezone()
        self.updated_at: datetime = datetime.now().astimezone()
        self.lead_time: float = 0
        self.task: int = 0
        self.results = Results()  # or Dict?


def submit_annotations(results: Dict, project_id: int, users_id: int, task_id: int, annotation_id: int, is_labelled: bool = True, conn=conn) -> int:
    """ Submit results for new annotations

    Args:
        results (Dict): [description]
        project_id (int): [description]
        users_id (int): [description]
        task_id (int): [description]
        annotation_id (int): [description]
        is_labelled (bool, optional): [description]. Defaults to True.
        conn (psycopg2 connection object, optional): [description]. Defaults to conn.

    Returns:
        [type]: [description]
    """

    # TODO is it neccessary to have annotation type id?
    insert_annotations_SQL = """
                                INSERT INTO public.annotations (
                                    results,
                                    project_id,
                                    users_id,
                                    task_id)
                                VALUES (
                                    %s::jsonb,
                                    %s,
                                    %s, 
                                    %s) 
                                RETURNING id;
                            """, [json.dumps(results), project_id, users_id, task_id]
    annotation_id = db_fetchone(insert_annotations_SQL, conn)

    update_task_SQL = """
                        UPDATE
                            public.task
                        SET
                            (annotation_id = %s),
                            (is_labelled = %s)
                        WHERE
                            id = %s;
                    """
    context = [annotation_id, is_labelled, task_id]
    db_no_fetch(update_task_SQL, conn, context)

    return annotation_id


def update_annotations(results: Dict, users_id: int, annotation_id: int, conn=conn) -> tuple:
    """Update results for new annotations

    Args:
        results (Dict): [description]
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
                                    (results = %s::jsonb),
                                    (users_id = %s)
                                WHERE
                                    id = %s
                                RETURNING *;
                            """, [json.dumps(results), users_id]
    updated_annotation_return = db_fetchone(update_annotations_SQL, conn)

    return updated_annotation_return


def skip_task(task_id: int, skipped: bool) -> tuple:
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
                """, [skipped, task_id]

    skipped_task_return = db_fetchone(skip_task_SQL, conn)

    return skipped_task_return


def delete_annotation(annotation_id: int) -> tuple:
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
                            """, [annotation_id]

    delete_annotation_return = db_fetchone(delete_annotations_SQL, conn)

    return delete_annotation_return
