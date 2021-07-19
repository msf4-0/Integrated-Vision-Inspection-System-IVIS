"""
Title: Dataset Management
Date: 18/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
from typing import Union, List
import psycopg2
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as SessionState

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[3]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
TEST_MODULE_PATH = SRC / "test" / "test_page" / "module"

for path in sys.path:
    if str(LIB_PATH) not in sys.path:
        sys.path.insert(0, str(LIB_PATH))  # ./lib
    else:
        pass

    if str(TEST_MODULE_PATH) not in sys.path:
        sys.path.insert(0, str(TEST_MODULE_PATH))
    else:
        pass
# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
from data_manager.database_manager import init_connection, db_fetchone
from core.utils.file_handler import bytes_divisor
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])
# >>>> Variable Declaration <<<<

# <<<< Variable Declaration <<<<


class BaseDataset:
    def __init__(self, dataset_id) -> None:
        self.dataset_id = dataset_id
        self.name: str = None
        self.desc: str = None
        self.file_type: str = None
        self.dataset_size: int = None  # Number of files
        self.dataset_path: str = None
        self.deployment_id: Union[str, int] = None
        self.deployment_type: str = None
        self.dataset = []


class NewDataset(BaseDataset):
    def __init__(self, dataset_id) -> None:
        # init BaseDataset -> Temporary dataset ID from random gen
        super().__init__(dataset_id)
        self.dataset_total_filesize = 0  # in byte-size
        self.has_submitted = False

    def query_deployment_id(self) -> int:
        query_id_SQL = """
                        SELECT
                            id
                        FROM
                            public.deployment_type
                        WHERE
                            name = %s;
                        """
        if self.deployment_type is not None and self.deployment_type != '':

            self.deployment_id = db_fetchone(
                query_id_SQL, [self.deployment_type], conn)[0]
        else:
            self.deployment_id = None

    def create_dataset_directory(self):
        print("Hi")

    def check_if_field_empty(self, field: List, field_placeholder):
        empty_fields = []
        keys = ["name", "deployment_type", "upload"]
        # if not all_field_filled:  # IF there are blank fields, iterate and produce error message
        for i in field:
            if i:
                pass

            else:
                idx = field.index(i)
                field_placeholder[keys[idx]].error(
                    f"Please do not leave field blank")
                empty_fields.append(keys[idx])

        return not empty_fields

    def calc_total_filesize(self):
        if self.dataset:
            self.dataset_total_filesize = 0
            for data in self.dataset:
                self.dataset_total_filesize += data.size
            # To get size in MB
            self.dataset_total_filesize = bytes_divisor(
                self.dataset_total_filesize, -2)
        return self.dataset_total_filesize

    def check_if_exist(self):
        check_exist_SQL = """
                            SELECT
                                EXISTS (
                                    SELECT
                                        name
                                    FROM
                                        public.dataset
                                    WHERE
                                        name = %s);
                        """
        exist_status=db_fetchone(check_exist_SQL,self.name)
    def insert_dataset(self):
        insert_dataset_SQL = """
                                INSERT INTO public.dataset (
                                    name,
                                    description,
                                    file_type,
                                    dataset_path,
                                    dataset_size,
                                    deployment_id)
                                VALUES (
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s)
                                RETURNING
                                    id;
                            """
        insert_dataset_vars = [self.name, self.desc,
                               self.file_type, self.file_type, self.dataset_path, self.dataset_size, self.deployment_id]
        self.dataset_id = db_fetchone(
            insert_dataset_SQL, insert_dataset_vars, conn)[0]
        return self.dataset_id

    def save_dataset(self):
        


def main():
    print("Hi")


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
