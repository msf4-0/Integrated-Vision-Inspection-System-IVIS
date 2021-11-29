"""
Title: Deployment Management
Date: 28/7/2021
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

from collections import namedtuple
import sys
from pathlib import Path
from typing import NamedTuple, Optional, Union, List, Dict
from psycopg2 import sql
from PIL import Image
from time import sleep
from enum import IntEnum
import json
from copy import copy, deepcopy
import pandas as pd
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
from path_desc import chdir_root, MEDIA_ROOT
from core.utils.log import log_info, log_error  # logger
from core.utils.helper import get_identifier_str_IntEnum
from data_manager.database_manager import init_connection, db_fetchone, db_no_fetch, db_fetchall
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration >>>>

# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


class DeploymentType(IntEnum):
    Image_Classification = 1
    OD = 2
    Instance = 3
    Semantic = 4

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return DeploymentType[s]
        except KeyError:
            raise ValueError()


# KIV
DEPLOYMENT_TYPE = {
    "Image Classification": DeploymentType.Image_Classification,
    "Object Detection with Bounding Boxes": DeploymentType.OD,
    "Semantic Segmentation with Polygons": DeploymentType.Instance,
    "Semantic Segmentation with Masks": DeploymentType.Semantic
}

COMPUTER_VISION_LIST = [DeploymentType.Image_Classification, DeploymentType.OD,
                        DeploymentType.Instance, DeploymentType.Semantic]
# <<<< Variable Declaration <<<<

# **************************************************************************************************************
# ********************************************** OUTDATED ******************************************************
# ******************************************** REQUIRES UPDATE ************************************************
# **************************************************************************************************************


class BaseDeployment:
    def __init__(self) -> None:
        self.id: int = None
        self.name: str = None
        self.model_selected = None


class Deployment(BaseDeployment):
    def __init__(self) -> None:
        super().__init__()
        self.deployment_list: List = self.query_deployment_list()

    @st.cache
    def query_deployment_list(self):
        query_deployment_list_sql = """
                                    SELECT
                                        name
                                    FROM
                                        deployment_type
                                    ORDER BY
                                        id ASC;
                                    """
        deployment_list = db_fetchall(query_deployment_list_sql, conn)
        return deployment_list if deployment_list else None

    @st.cache
    def query_model_table(self, model_table) -> NamedTuple:
        schema, table = [x for x in model_table.split('.')]
        query_model_table_SQL = sql.SQL("""SELECT
                m.id AS "ID",
                m.name AS "Name",
                f.name AS "Framework",
                m.model_path AS "Model Path"
            FROM
                {table} m
                LEFT JOIN public.framework f ON f.id = m.framework_id
                where m.deployment_id = (SELECT id from public.deployment_type where name = %s);""").format(table=sql.Identifier(schema, table))
        query_model_table_vars = [self.name]
        return_all = db_fetchall(
            query_model_table_SQL, conn, query_model_table_vars, fetch_col_name=True)
        if return_all:
            project_model_list, column_names = return_all
        else:
            project_model_list = []
            column_names = []
        return project_model_list, column_names

    @staticmethod
    @st.experimental_memo
    def get_deployment_type(deployment_type: Union[str, DeploymentType], string: bool = False):

        assert isinstance(
            deployment_type, (str, DeploymentType)), f"deployment_type must be String or IntEnum"

        deployment_type = get_identifier_str_IntEnum(
            deployment_type, DeploymentType, DEPLOYMENT_TYPE, string=string)

        return deployment_type


def main():
    print("Hi")


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
