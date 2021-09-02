"""
Title: Model Management
Date: 20/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

from collections import namedtuple
import sys
from pathlib import Path
from typing import NamedTuple, Tuple, Union, List, Dict
import pandas as pd
import psycopg2
from PIL import Image
from time import sleep
from enum import IntEnum
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
from path_desc import chdir_root, MEDIA_ROOT, PROJECT_DIR, PRE_TRAINED_MODEL_DIR, USER_DEEP_LEARNING_MODEL_UPLOAD
from core.utils.log import log_info, log_error  # logger
from core.utils.helper import get_dataframe_row, get_directory_name
from data_manager.database_manager import db_fetchall, init_connection, db_fetchone, db_no_fetch
from core.utils.helper import create_dataframe, dataframe2dict, get_directory_name, datetime_formatter, get_identifier_str_IntEnum
from core.utils.form_manager import check_if_exists, check_if_field_empty
from deployment.deployment_management import Deployment

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])

# >>>> Variable Declaration >>>>


class ModelType(IntEnum):
    PreTrained = 0
    ProjectTrained = 1
    UserUpload = 2

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return ModelType[s]
        except KeyError:
            raise ValueError()


MODEL_TYPE = {
    ModelType.PreTrained: "Pre-trained Models",
    ModelType.ProjectTrained: "Project Models",
    ModelType.UserUpload: "User Deep Learning Model Upload"
}


class Framework(IntEnum):
    TensorFlow = 0
    PyTorch = 1
    Scikit_learn = 2
    Caffe = 3
    MXNet = 4
    ONNX = 5

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return Framework[s]
        except KeyError:
            raise ValueError()


FRAMEWORK = {
    Framework.TensorFlow: "TensorFlow",
    Framework.PyTorch: "PyTorch",
    Framework.Scikit_learn: "Scikit-learn",
    Framework.Caffe: "Caffe",
    Framework.MXNet: "MXNet",
    Framework.ONNX: "ONNX"
}
# <<<< Variable Declaration <<<<

# >>>> TODO >>>>


class BaseModel:
    def __init__(self, model_id: Union[int, str]) -> None:
        self.id: Union[str, int] = model_id
        self.name: str = None
        self.desc: str = None
        self.metrics: Dict = {}
        self.model_type: str = None
        self.framework: str = None
        self.training_id: int = None
        self.model_path: Path = None
        self.labelmap_path: Path = None
        self.saved_model_dir: Path = None
        self.has_submitted: bool = False

    # TODO Method to generate Model Path #116
    @st.cache
    def get_model_path(self):
        query_model_project_training_SQL = """
                SELECT
                    p.project_path,
                    t.name
                FROM
                    public.models m
                    INNER JOIN public.training t ON m.training_id = t.id
                    INNER JOIN public.project p ON t.project_id = p.id
                WHERE
                    m.id = %s;
                        """
        query_model_project_training_vars = [self.id]
        query = db_fetchone(query_model_project_training_SQL,
                            conn, query_model_project_training_vars)

        return query

    # Wrapper for check_if_exists function from form_manager.py
    def check_if_exists(self, context: List, conn) -> bool:
        table = 'public.models'
        exists_flag = check_if_exists(
            table, context['column_name'], context['value'], conn)

        return exists_flag

    # Wrapper for check_if_exists function from form_manager.py
    def check_if_field_empty(self, context: Dict, field_placeholder):
        check_if_exists = self.check_if_exists
        empty_fields = check_if_field_empty(
            context, field_placeholder, check_if_exists)
        return empty_fields

    @staticmethod
    def get_model_type(model_type: Union[str, ModelType], string: bool = False):

        assert isinstance(
            model_type, (str, ModelType)), f"deployment_type must be String or IntEnum"

        model_type = get_identifier_str_IntEnum(
            model_type, ModelType, MODEL_TYPE, string=string)

        return model_type

    @staticmethod
    def get_framework(framework: Union[str, ModelType], string: bool = False):

        assert isinstance(
            framework, (str, Framework)), f"deployment_type must be String or IntEnum"

        model_type = get_identifier_str_IntEnum(
            framework, Framework, FRAMEWORK, string=string)

        return model_type

    def get_pt_user_model_path(model_path: Union[str, Path] = None, framework: str = None, model_type: Union[str, ModelType] = None, **model_row) -> Path:
        """Get directory path for Pre-trained models and User Upload Deep Learning Models

        Args:
            model_path (Union[str, Path]): Relative path to model (queried from DB)
            framework (str): Framework of model
            model_type (Union[str, ModelType]): Type of model

        Returns:
            Path: Path-like object of model path
        """

        if model_row:
            model_path = model_row['Model Path']
            framework = model_row['Framework']
            model_type = model_row['Model Type']

        # Get IntEnum constant
        model_type = BaseModel.get_model_type(
            model_type=model_type, string=False)

        framework = get_directory_name(framework)

        if model_type == ModelType.PreTrained:
            model_path = PRE_TRAINED_MODEL_DIR / framework / model_path

        elif model_type == ModelType.UserUpload:
            model_path = USER_DEEP_LEARNING_MODEL_UPLOAD / framework / model_path

        # assert model_path.is_dir(), f"{str(model_path)} does not exists"

        if not model_path.is_dir():
            error_msg = f"{str(model_path)} does not exists"
            log_error(error_msg)
            st.error(error_msg)
            model_path = None

        return model_path

    @staticmethod
    def query_project_model_path(model_id: int):
        query_model_project_training_SQL = """
            SELECT m.name AS "Name",
                (SELECT p.name AS "Project Name"
                    FROM public.project p
                            INNER JOIN project_training pt ON p.id = pt.project_id
                    WHERE m.training_id = pt.training_id),
                (SELECT t.name AS "Training Name" FROM public.training t WHERE m.training_id = t.id),
                (
                    SELECT f.name AS "Framework"
                    FROM public.framework f
                    WHERE f.id = m.framework_id
                )

            FROM public.models m
            WHERE m.id = %s;
                                    """
        query_model_project_training_vars = [model_id]
        query_result = db_fetchone(query_model_project_training_SQL,
                                   conn, query_model_project_training_vars)
        if query_result:

            project_model_path = PROJECT_DIR / \
                get_directory_name(query_result.Project_Name) / get_directory_name(
                    query_result.Training_Name) / get_directory_name(query_result.Framework) /\
                'exported_models' / query_result.Model_Path
            return project_model_path


class NewModel(BaseModel):
    def __init__(self, model_id: str) -> None:
        super().__init__(model_id)


class Model(BaseModel):
    def __init__(self, model_id: int) -> None:
        super().__init__(model_id)

    @staticmethod
    def query_model_table(for_data_table: bool = False, return_dict: bool = False, deployment_type: Union[str, IntEnum] = None) -> namedtuple:
        """Wrapper function to query model table

        Args:
            for_data_table (bool, optional): True if query for data table. Defaults to False.
            return_dict (bool, optional): True if query results of type Dict. Defaults to False.
            deployment_type (IntEnum, optional): Deployment type. Defaults to None.

        Returns:
            namedtuple: [description]
        """
        if deployment_type:
            models, column_names = query_model_ref_deployment_type(deployment_type=deployment_type,
                                                                   for_data_table=for_data_table,
                                                                   return_dict=return_dict)

        else:
            models, column_names = query_all_models(
                for_data_table=for_data_table, return_dict=return_dict)

        return models, column_names

    @staticmethod
    @st.cache
    def get_framework_list() -> List[namedtuple]:
        """Get list of Deep Learning frameworks from Database

        Returns:
            List[namedtuple]: List of framework in namedtuple (ID, Name)
        """
        get_framework_list_SQL = """
            SELECT
                id as "ID",
                name as "Name"
            FROM
                public.framework;
                    """
        framework_list = db_fetchall(get_framework_list_SQL, conn)
        return framework_list

    @staticmethod
    def create_models_dataframe(models: Union[List[namedtuple], List[dict]],
                                column_names: List = None, sort_col: str = None
                                ) -> pd.DataFrame:
        """Generate Pandas DataFrame to store Models query

        Args:
            models (Union[List[namedtuple], List[dict]]): Models query from 'query_model_table()'
            column_names (List, optional): Names of columns. Defaults to None.
            sort_col (str, optional): Sort value. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame of Models query
        """
        df = create_dataframe(models, column_names,
                              date_time_format=True, sort_by=sort_col, asc=True)

        df['Date/Time'] = df['Date/Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        return df

    @staticmethod
    @dataframe2dict(orient='index')
    def filtered_models_dataframe(models: Union[List[namedtuple], List[dict]],
                                  dataframe_col: str, filter_value: Union[str, int],
                                  column_names: List = None, sort_col: str = None
                                  ) -> pd.DataFrame:
        """Get a List of filtered Models Dict using pandas.DataFrame.loc[]

        Args:
            models (Union[List[namedtuple], List[dict]]): models query from 'query_models_table'
            dataframe_col (str): DataFrame column to be filtered
            filter_value (Union[str, int]): Filter attribute

        Returns:
            List[Dict]: Filtered DataFrame
        """

        models_df = Model.create_models_dataframe(
            models, column_names, sort_col=sort_col)
        filtered_models_df = models_df.loc[models_df[dataframe_col]
                                           == filter_value]

        return filtered_models_df

    def get_project_model_path(self):
        self.model_path = BaseModel.query_project_model_path(self.id)
        return self.model_path

    def get_labelmap_path(self):
        model_path = self.get_model_path()
        if model_path:
            labelmap_path = model_path / 'labelmap.pbtxt'
            self.labelmap_path = labelmap_path

            return self.labelmap_path

    def get_model_row(model_id: int, model_df: pd.DataFrame) -> Dict:
        """Get selected model row

        Args:
            model_id (int): Model ID
            model_df (pd.DataFrame): DataFrame for models

        Returns:
            Dict: Data row from models DataFrame
        """
        log_info(f"Obtaining data row from model_df......")

        model_row = get_dataframe_row(model_id, model_df)

        log_info(f"Currently serving data:{model_row['Name']}")

        return model_row


class PreTrainedModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.pt_model_list, self.pt_model_column_names = self.query_PT_table()

    # DEPRECATED?
    @st.cache
    def query_PT_table(self) -> NamedTuple:
        query_PT_table_SQL = """
            SELECT
                pt.id AS "ID",
                pt.name AS "Name",
                f.name AS "Framework",
                dt.name AS "Deployment Type",
                pt.model_path AS "Model Path"
            FROM
                public.pre_trained_models pt
                LEFT JOIN public.framework f ON f.id = pt.framework_id
                LEFT JOIN public.deployment_type dt ON dt.id = pt.deployment_id;"""
        PT_model_list, column_names = db_fetchall(
            query_PT_table_SQL, conn, fetch_col_name=True)
        return PT_model_list, column_names


def query_all_models(for_data_table: bool = False, return_dict: bool = False):

    ID_string = "id" if for_data_table else "ID"

    query_all_model_SQL = f"""
            SELECT
                m.id AS \"{ID_string}\",
                m.name AS "Name",
                (
                    SELECT
                        f.name AS "Framework"
                    FROM
                        public.framework f
                    WHERE
                        f.id = m.framework_id
                ),
                (
                    SELECT
                        mt.name AS "Model Type"
                    FROM
                        public.model_type mt
                    WHERE
                        mt.id = m.model_type_id
                ),
                (
                    SELECT
                        dt.name AS "Deployment Type"
                    FROM
                        public.deployment_type dt
                    WHERE
                        dt.id = m.deployment_id
                ),
                (
                    /* Replace NULL with '-' */
                    SELECT
                        CASE
                            WHEN m.training_id IS NULL THEN '-'
                            ELSE (
                                SELECT
                                    t.name
                                FROM
                                    public.training t
                                WHERE
                                    t.id = m.training_id
                            )
                        END AS "Training Name"
                ),
                m.description AS "Description",
                m.metrics AS "Metrics",
                m.model_path AS "Model Path"
            FROM
                public.models m
            ORDER BY
                ID ASC;
                    """

    models, column_names = db_fetchall(
        query_all_model_SQL, conn, fetch_col_name=True, return_dict=return_dict)

    log_info(f"Querying all models......")

    models_tmp = []

    if models:
        models_tmp = datetime_formatter(models, return_dict=return_dict)

    else:
        models_tmp = []

    return models_tmp, column_names


def query_model_ref_deployment_type(deployment_type: Union[str, IntEnum] = None, for_data_table: bool = False, return_dict: bool = False):
    """Query rows of models filtered by Deployment Type from 'models' table

    Args:
        deployment_type (str, optional): [description]. Defaults to None.
        for_data_table (bool, optional): [description]. Defaults to False.
        return_dict (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    deployment_type = Deployment.get_deployment_type(
        deployment_type=deployment_type, string=True)
    ID_string = "id" if for_data_table else "ID"

    model_dt_SQL = f"""
        SELECT
            m.id AS \"{ID_string}\",
            m.name AS "Name",
            (
                SELECT
                    f.name AS "Framework"
                FROM
                    public.framework f
                WHERE
                    f.id = m.framework_id
            ),
            (
                SELECT
                    mt.name AS "Model Type"
                FROM
                    public.model_type mt
                WHERE
                    mt.id = m.model_type_id
            ),
            (
                /* Replace NULL with '-' */
                SELECT
                    CASE
                        WHEN m.training_id IS NULL THEN '-'
                        ELSE (
                            SELECT
                                t.name
                            FROM
                                public.training t
                            WHERE
                                t.id = m.training_id
                        )
                    END AS "Training Name"
            ),
            --        m.updated_at  AS "Date/Time",
            m.description AS "Description",
            m.metrics AS "Metrics",
            m.model_path AS "Model Path"
        FROM
            public.models m
            INNER JOIN public.deployment_type dt ON dt.name = 'Object Detection with Bounding Boxes'
        ORDER BY
            m.id ASC;        
                """

    model_dt_vars = [deployment_type]

    models, column_names = db_fetchall(
        model_dt_SQL, conn, model_dt_vars, fetch_col_name=True, return_dict=return_dict)

    log_info(f"Querying models filtered by Deployment Type from database....")

    models_tmp = []

    if models:
        models_tmp = datetime_formatter(models, return_dict)

    else:
        models_tmp = []

    return models_tmp, column_names
