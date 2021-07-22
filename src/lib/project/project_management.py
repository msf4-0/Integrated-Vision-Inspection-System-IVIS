"""
Title: User Management
Date: 25/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
from typing import Union, List, Dict
import psycopg2
from PIL import Image
from time import sleep
from enum import IntEnum
from copy import copy, deepcopy
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

# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
from data_manager.database_manager import init_connection, db_fetchone, db_no_fetch, db_fetchall
from core.utils.file_handler import bytes_divisor, create_folder_if_not_exist
from core.utils.helper import split_string, join_string
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration >>>>

# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


class AccountStatus(IntEnum):
    # **** User Status ****

    NEW = 0  # Pending account activation
    ACTIVE = 1  # Account activated
    LOCKED = 2  # Account locked
    LOGGED_IN = 3  # Account logged-in
    LOGGED_OUT = 4  # Account logged-out

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return AccountStatus[s]
        except KeyError:
            raise ValueError()

# <<<< Variable Declaration <<<<


# >>>> TODO >>>>
ACTIVATION_FUNCTION = ['RELU_6', 'sigmoid']
OPTIMIZER = []
CLASSIFICATION_LOSS = []


class TrainingParam:
    def __init__(self) -> None:
        self.epoch: Union[int, float] = 50
        self.batch_size: int = 2
        self.learning_rate: Union[int, float] = 0.0008
        self.total_steps: int = 50000
        self.warmup_steps = 1000
        self.warmup_learning_rate = 0.00001
        self.shuffle_buffer_size = 2048
        self.activation_function: str = 'RELU_6'
        self.optimizer: str = None
        self.classification_loss: str = 'weighted_sigmoid_focal'

# >>>> TODO >>>>


class Augmentation:
    def __init__(self) -> None:
        pass


class BaseTraining:
    def __init__(self, training_id) -> None:
        self.id: Union[str, int] = training_id
        self.name: str = None
        self.training_param: TrainingParam = TrainingParam()
        self.augmentation: Augmentation = Augmentation()  # TODO


class NewTraining(BaseTraining):
    def __init__(self, training_id) -> None:
        super().__init__(training_id)


class BaseProject:
    def __init__(self, project_id) -> None:
        self.id = project_id
        self.name: str = None
        self.desc: str = None
        self.project_path: Path = None
        self.deployment_id: Union[str, int] = None
        self.dataset_id: int = None
        self.training_id: int = None
        self.editor_config: str = None
        self.deployment_type: str = None
        self.dataset_chosen: List = []
        self.project = []  # keep?
        self.project_size: int = None  # Number of files
        self.datasets: List = self.query_dataset_list()
        self.dataset_name_list: List = self.get_dataset_name_list()
        self.dataset_name_id: Dict = {}

    @st.cache
    def query_dataset_list(self) -> List:
        query_dataset_SQL = """SELECT
                                id,
                                name,
                                dataset_size,
                                updated_at
                            FROM
                                public.dataset;"""

        datasets = db_fetchall(query_dataset_SQL, conn)
        dataset_tmp = []
        if datasets:
            for dataset in datasets:
                dataset = list(dataset)  # convert tuples to List

                # convert datetime with TZ to (2021-07-30 12:12:12) format
                dataset[3] = dataset[3].strftime('%Y-%m-%d %H:%M:%S')
                dataset_tmp.append(dataset)

            self.datasets = dataset_tmp
        else:
            dataset_tmp = []

        return dataset_tmp

    def get_dataset_name_list(self) -> List:
        dataset_name_tmp = []
        dataset_name_id = {}
        if self.datasets:
            for dataset in self.datasets:
                dataset_name_tmp.append(dataset[1])
                dataset_name_id[dataset[1]] = dataset[0]
            self.dataset_name_list = dataset_name_tmp
            self.dataset_name_id = dataset_name_id
        else:
            self.dataset_name_list = []
            self.dataset_name_id = {}

        return dataset_name_tmp, dataset_name_id

    def create_dataset_dataframe(self) -> pd.DataFrame:

        if self.datasets:
            df = pd.DataFrame(self.datasets, columns=[
                'ID', 'Name', 'Dataset Size', 'Date/Time'])
            df['Date/Time'] = pd.to_datetime(df['Date/Time'],
                                             format='%Y-%m-%d %H:%M:%S')
            df.sort_values(by=['Date/Time'], inplace=True,
                           ascending=False, ignore_index=True)
            df.index.name = ('No.')

            # dfStyler = df.style.set_properties(**{'text-align': 'center'})
            # dfStyler.set_table_styles(
            #     [dict(selector='th', props=[('text-align', 'center')])])

        return df


class NewProject(BaseProject):
    def __init__(self, project_id) -> None:
        # init BaseDataset -> Temporary dataset ID from random gen
        super().__init__(project_id)
        self.project_total_filesize = 0  # in byte-size
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
                query_id_SQL, conn, [self.deployment_type])[0]
        else:
            self.deployment_id = None


# TODO


    def check_if_field_empty(self, field: List, field_placeholder):
        empty_fields = []
        keys = ["name", "deployment_type", "dataset_chosen"]
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

    def check_if_exist(self, context: List, conn) -> bool:
        check_exist_SQL = """
                            SELECT
                                EXISTS (
                                    SELECT
                                        %s
                                    FROM
                                        public.project
                                    WHERE
                                        name = %s);
                        """
        exist_status = db_fetchone(check_exist_SQL, conn, context)[0]
        return exist_status

    def insert_project(self):
        insert_project_SQL = """
                                INSERT INTO public.project (
                                    name,
                                    description,                                    
                                    project_path,                                    
                                    deployment_id)
                                VALUES (
                                    %s,
                                    %s,
                                    %s,
                                    %s)
                                RETURNING id;
                                
                            """
        insert_project_vars = [self.name, self.desc,
                               str(self.project_path), self.deployment_id]
        self.id = db_fetchone(
            insert_project_SQL, conn, insert_project_vars)[0]
        insert_project_dataset_SQL = """
                                        INSERT INTO public.project_dataset (
                                            project_id,
                                            dataset_id)
                                        VALUES (
                                            %s,
                                            %s);"""
        for dataset in self.dataset_chosen:
            dataset_id = self.dataset_name_id[dataset]
            insert_project_dataset_vars = [self.id, dataset_id]
            db_no_fetch(insert_project_dataset_SQL, conn,
                        insert_project_dataset_vars)
        return self.id

    # def insert_project_dataset(self):

    #     insert_project_dataset_SQL = """
    #                                     INSERT INTO public.project_dataset (
    #                                         project_id,
    #                                         dataset_id)
    #                                     VALUES (
    #                                         %s,
    #                                         %s);"""
    #     for dataset in self.dataset_chosen:
    #         dataset_id = self.dataset_name_id[dataset]
    #         insert_project_dataset_vars = [self.id, dataset_id]
    #         db_no_fetch(insert_project_dataset_SQL, conn,
    #                     insert_project_dataset_vars)

    def initialise_project(self):
        directory_name = self.name.lower()
        directory_name = join_string(split_string(str(directory_name)))
        self.project_path = Path.home() / '.local' / 'share' / \
            'integrated-vision-inspection-system' / \
            'app_media' / 'project' / str(directory_name)
        create_folder_if_not_exist(self.project_path)
        log_info(
            f"Successfully created **{self.name}** project at {str(self.project_path)}")
        if self.insert_project():

            log_info(
                f"Successfully stored **{self.name}** project information in database")
            return True

        else:
            log_error(
                f"Failed to stored **{self.name}** project information in database")
            return False


# TODO: move to form_manager


def check_if_field_empty(new_user, field_placeholder, field_name):
    empty_fields = []
    # all_field_filled = all(new_user)
    # if not all_field_filled:  # IF there are blank fields, iterate and produce error message
    for key, value in new_user.items():
        if value == "":
            field_placeholder[key].error(
                f"Please do not leave {field_name[key]} field blank")
            empty_fields.append(key)

        else:
            pass

    return not empty_fields


# TODO:KIV can be removed


def create_project_table(conn=conn):  # Create Table
    # create relation : user_details
    create_project_table_SQL = """
                                CREATE TABLE IF NOT EXISTS public.project (
                                id bigint NOT NULL GENERATED BY DEFAULT AS IDENTITY (INCREMENT 1 START 1
                                MINVALUE 1
                                MAXVALUE 9223372036854775807
                                CACHE 1),
                                name text NOT NULL UNIQUE,
                                description text,
                                deployment_id integer,
                                dataset_id bigint,
                                training_id bigint,
                                editor_config text,
                                created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
                                updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
                                PRIMARY KEY (id),
                                CONSTRAINT fk_deployment_id FOREIGN KEY (deployment_id) REFERENCES public.deployment_type (id) ON DELETE SET NULL,
                                CONSTRAINT fk_training_id FOREIGN KEY (training_id) REFERENCES public.training (id) ON DELETE SET NULL)
                            TABLESPACE image_labelling;

                            CREATE TRIGGER project_update
                                BEFORE UPDATE ON public.project
                                FOR EACH ROW
                                EXECUTE PROCEDURE trigger_update_timestamp ();

                            ALTER TABLE public.project OWNER TO shrdc;
                            """
    db_no_fetch(create_project_table_SQL, conn)

# >>>> CREATE PROJECT >>>>


def main():
    print("Hi")


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
