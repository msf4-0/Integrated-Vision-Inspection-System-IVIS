"""
Title: Annotation Manager
Date: 15/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
from typing import List, Dict, Union
from enum import IntEnum
from datetime import datetime
import psycopg2
from psycopg2.extras import Json
import json
from base64 import b64encode
from PIL import Image
from io import BytesIO
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
from user.user_management import User
from core.utils.dataset_handler import data_url_encoder_cv2
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
                                RETURNING id;
                                        """
        insert_new_task_vars = [image_name, project_id, dataset_id]
        task_id = db_fetchone(insert_new_task_SQL, conn,
                              insert_new_task_vars).id

        return task_id


class Task(BaseTask):
    def __init__(self, data_object, data_name, project_id, dataset_id, annotations=None, predictions=None) -> None:
        super().__init__()
        self.data = {'image': None}  # generate when call LS format generator
        self.name = data_name
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.annotations = annotations
        self.predictions = predictions
        self.data_object = data_object
        self.query_task()

    # TODO: check if current image exists as a 'task' in DB
    @staticmethod
    def check_if_task_exists(image_name: str, project_id: int, dataset_id: int, conn=conn) -> bool:
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
            log_error(
                f"{e}: Task for data {self.name} from Dataset {self.dataset_id} does not exist in table for Project {self.project_id}")

    def generate_data_url(self):
        """Generate data url from OpenCV numpy array

        Returns:
            str: Data URl with base64 bytes encoded in UTF-8
        """

        try:
            data_url = data_url_encoder_cv2(self.data_object, self.name)

            return data_url
        except Exception as e:
            log_error(f"{e}: Failed to generate data url for {self.name}")
            return None

    @st.cache
    def get_data(self):

        data = self.data_list.get(self.name)

        if not data:
            data = self.generate_data_url()
            # add encoded image into dictionary
            self.data_list[self.name] = data

        else:
            log_info(f"Data {self.name} EXIST")

        # if self.name in self.data_list.keys():
        #     data = self.data_list[self.name]
        #     log_info(f"Data {self.name} EXIST")
        # else:
        #     data = self.generate_data_url()
        #     # add encoded image into dictionary
        #     self.data_list[self.name] = data
        return data

    def generate_editor_format(self, annotations_dict, predictions_dict=None):
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
                "data": {"image": self.get_data()},
                # 'data': {'image': None}
                # 'meta': {},
                # 'created_at': str(self.created_at),
                # 'updated_at': str(self.updated_at),
                # 'project': self.project_id
            }
            return task
        except Exception as e:
            log_error(f"{e}: Failed to generate editor format for editor")
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
            log_error(f"{error}: Annotations already exist")

        # NOTE: Update 'task' table with annotation id and set is_labelled flag as True
        update_task_SQL = """
                            UPDATE
                                public.task
                            SET
                                annotation_id = %s,
                                is_labelled = %s
                            WHERE
                                id = %s;
                        """
        self.task.is_labelled = True
        update_task_vars = [self.id,
                            self.task.is_labelled, self.task.id]
        db_no_fetch(update_task_SQL, conn, update_task_vars)

        return self.id

    def update_annotations(self, result: Dict, users_id: int, conn=conn) -> tuple:
        """Update result for new annotations

        Args:
            result (Dict): [description]
            users_id (int): [description]
            conn (psycopg2 connection object, optional): [description]. Defaults to conn.

        Returns:
            tuple: [description]
        """
        self.result = result if result else None  # update result attribute
        result_serialised = [json.dumps(x) for x in result]
        # TODO is it neccessary to have annotation type id?
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
            log_info(
                f"Update annotations for Task {self.task.name} with Annotation ID: {self.id}")
            return updated_annotation_return
        except psycopg2.Error as e:
            error = e.pgcode
            log_error(f"{error}: Annotations already exist")

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
                            skipped = %s
                        WHERE
                            id = %s
                        RETURNING *;
                    """
        skip_task_vars = [
            skipped, self.task.id]  # should set 'skipped' as True
        log_info(self.task.id)
        skipped_task_return = db_fetchone(skip_task_SQL, conn, skip_task_vars)

        return skipped_task_return

    def generate_annotation_dict(self) -> Union[Dict, List]:
        try:
            if self.task.is_labelled:
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
            log_error(
                f"{e}: Failed to generate annotation dict for Annotation {self.id}")
            annotation_dict = {}
            return annotation_dict


class NewAnnotations(BaseAnnotations):
    def __init__(self, task: Task) -> None:
        super().__init__(task)

    def generate_annotation_dict(self) -> Union[Dict, List]:
        try:
            annotation_dict = []
            return annotation_dict
        except Exception as e:
            log_error(
                f"{e}: Failed to generate annotation dict for Annotation {self.id}")
            annotation_dict = []
            return annotation_dict


class Annotations(BaseAnnotations):
    def __init__(self, task: Task) -> None:
        super().__init__(task)
        log_info(f"Initialising New Task {self.task.name}")
        self.task = task  # 'Task' class object
        self.user = {}
        if self.task.is_labelled:
            self.query_annotations()
            self.completed_by = {
                "id": self.user["id"], "email": self.user["email"], "first_name": self.user["first_name"], "last_name": self.user["last_name"]}
        else:
            pass

    # TODO: check if current image exists as a 'task' in DB

    @staticmethod
    def check_if_annotation_exists(task_id: int, project_id: int, conn=conn) -> bool:
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

    def query_annotations(self):
        """Query ID and Result

        Returns:
            [type]: [description]
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
                                inner join public.users u
                                on a.users_id = u.id
                                WHERE
                                    task_id = %s;

                                """
        query_annotation_vars = [self.task.id]
        query_return = db_fetchone(
            query_annotation_SQL, conn, query_annotation_vars, fetch_col_name=False)
        try:
            # self.user is NamedTuple
            self.id, self.result, self.user["id"], self.user["email"], self.user[
                "first_name"], self.user["last_name"], self.created_at, self.updated_at = query_return
            log_info("Query in class")
            return query_return
        except TypeError as e:
            log_error(
                f"{e}: Annotation for Task {self.task.id} from Dataset {self.task.dataset_id} does not exist in table for Project {self.project_id}")

    # def generate_annotation_dict(self) -> Union[Dict, List]:
    #     try:
    #         annotation_dict = {"id": self.id,
    #                            "completed_by": self.completed_by,
    #                            "result": [self.result],
    #                            "was_cancelled": self.was_cancelled,
    #                            "ground_truth": self.ground_truth,
    #                            "created_at": str(self.created_at),
    #                            "updated_at": str(self.updated_at),
    #                            "lead_time": str(self.updated_at - self.created_at),
    #                            "prediction": {},
    #                            "result_count": 0,
    #                            "task": self.task.id
    #                            }
    #         return annotation_dict
    #     except Exception as e:
    #         log_error(
    #             f"{e}: Failed to generate annotation dict for Annotation {self.id}")
    #         annotation_dict = {}
    #         return annotation_dict
# ********************** External Function *************************


@st.cache
def data_url_encoder(image):
    """Load Image and generate Data URL in base64 bytes

    Args:
        image (bytes-like): BytesIO object

    Returns:
        bytes: UTF-8 encoded base64 bytes
    """
    log_info("Loading sample image")
    bb = image.read()
    b64code = b64encode(bb).decode('utf-8')
    data_url = 'data:' + image.type + ';base64,' + b64code

    return data_url


@st.cache
def load_buffer_image():
    """Load Image and generate Data URL in base64 bytes

    Args:
        image (bytes-like): BytesIO object

    Returns:
        bytes: UTF-8 encoded base64 bytes
    """
    chdir_root()  # ./image_labelling
    log_info("Loading Sample Image")
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
