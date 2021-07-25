"""
Title: Model Management
Date: 20/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

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
from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
from data_manager.database_manager import db_fetchall, init_connection, db_fetchone, db_no_fetch
from core.utils.file_handler import bytes_divisor, create_folder_if_not_exist
from core.utils.helper import split_string, join_string
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration >>>>

# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])

# <<<< Variable Declaration <<<<

# >>>> TODO >>>>


class BaseModel:
    def __init__(self,) -> None:
        self.id: Union[str, int] = None
        self.name: str = None
        self.framework: str = None
        self.training_id: int = None
        self.model_path: Path = None


class Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()

    @st.cache
    def query_model_table(self) -> NamedTuple:
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


class PreTrainedModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()

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
