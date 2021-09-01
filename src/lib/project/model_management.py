"""
Title: Model Management
Date: 20/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

from collections import namedtuple
import sys
from pathlib import Path
from typing import NamedTuple, Union, List, Dict
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
from path_desc import chdir_root, MEDIA_ROOT
from core.utils.log import log_info, log_error  # logger
from data_manager.database_manager import db_fetchall, init_connection, db_fetchone, db_no_fetch
from core.utils.file_handler import bytes_divisor, create_folder_if_not_exist
from core.utils.helper import get_directory_name
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
    ModelType.UserUpload: "User Custom Deep Learning Model Upload"
}
# <<<< Variable Declaration <<<<

# >>>> TODO >>>>


class BaseModel:
    def __init__(self,) -> None:
        self.id: Union[str, int] = None
        self.name: str = None
        self.framework: str = None
        self.training_id: int = None
        self.model_path: Path = None
        self.labelmap_path: Path = None
        self.saved_model_dir: Path = None

    @staticmethod
    def query_table(expression: str, column: str):
        query_table_SQL = """
        SELECT
            %s
        FROM
            %s;
        """
        query_table_vars = [expression, column]
        return_all = db_fetchall(query_table_SQL, conn, query_table_vars)
        return return_all

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


class Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.p_model_list, self.p_model_column_names = self.query_model_table()

    @st.cache(ttl=60)
    def query_model_table(self) -> namedtuple:
        query_model_table_SQL = """
            SELECT
                m.id AS "ID",
                m.name AS "Name",
                f.name AS "Framework",
                dt.name AS "Deployment Type",
                m.model_path AS "Model Path"
            FROM
                public.models m
                LEFT JOIN public.framework f ON f.id = m.framework_id
                LEFT JOIN public.deployment_type dt ON dt.id = m.deployment_id;"""
        return_all = db_fetchall(
            query_model_table_SQL, conn, fetch_col_name=True)
        if return_all:
            project_model_list, column_names = return_all
        else:
            project_model_list = None
            column_names = None
        return project_model_list, column_names

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

    def check_if_exist(self, context: List, conn) -> bool:
        check_exist_SQL = """
                            SELECT
                                EXISTS (
                                    SELECT
                                        %s
                                    FROM
                                        public.models
                                    WHERE
                                        name = %s);
                        """
        exist_status = db_fetchone(check_exist_SQL, conn, context)[0]
        return exist_status

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
        if query:
            project_path, training_name = query
            self.model_path = MEDIA_ROOT / \
                project_path / get_directory_name(
                    training_name) / 'exported_models' / get_directory_name(self.name)
            return self.model_path

    def get_labelmap_path(self):
        model_path = self.get_model_path()
        if model_path:
            labelmap_path = model_path / 'labelmap.pbtxt'
            self.labelmap_path = labelmap_path
            return self.labelmap_path


class PreTrainedModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.pt_model_list, self.pt_model_column_names = self.query_PT_table()

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


def main():
    print("Hi")


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
