"""
Title: Dataset Management
Date: 18/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

from collections import namedtuple
import sys
from pathlib import Path
import os
from typing import Dict, Tuple, Union, List
from PIL import Image
from time import sleep, perf_counter
from glob import glob, iglob
from datetime import datetime
from stqdm import stqdm
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
from path_desc import chdir_root, DATA_DIR
from core.utils.log import log_info, log_error  # logger
from data_manager.database_manager import init_connection, db_fetchone, db_fetchall
from core.utils.file_handler import bytes_divisor, create_folder_if_not_exist
from core.utils.helper import get_directory_name
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
        self.dataset_size: int = None  # Number of files
        self.dataset_path: Path = None
        self.deployment_id: Union[str, int] = None
        self.filetype: str = None
        self.deployment_type: str = None
        self.dataset = []  # to hold new data from upload
        self.dataset_list = []  # List of existing dataset
        self.dataset_total_filesize = 0  # in byte-size
        self.has_submitted = False

# NOTE DEPRECATED *************************
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

    def check_if_field_empty(self, field: List, field_placeholder, keys=["name", "upload"]):
        empty_fields = []

        # if not all_field_filled:  # IF there are blank fields, iterate and produce error message
        for i in field:
            if i and i != "":

                # Double check if Dataset name exists in DB
                if (field.index(i) == 0) and ('name' in keys):
                    context = ['name', field[0]]
                    if self.check_if_exist(context, conn):
                        field_placeholder[keys[0]].error(
                            f"Dataset name used. Please enter a new name")
                        log_error(
                            f"Dataset name used. Please enter a new name")
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
                                        public.dataset
                                    WHERE
                                        name = %s);
                        """
        exist_status = db_fetchone(check_exist_SQL, conn, context)[0]
        return exist_status

    def dataset_PNG_encoding(self):
        if self.dataset:
            for img in stqdm(self.dataset, unit=self.filetype, ascii='123456789#'):
                img_name = img.name
                log_info(img.name)
                save_path = Path(self.dataset_path) / str(img_name)
                st.title(img.name)
                try:
                    with Image.open(img) as pil_img:
                        pil_img.save(save_path)
                except ValueError as e:
                    log_error(
                        f"{e}: Could not reolve output format for '{str(img_name)}'")
                except OSError as e:
                    log_error(
                        f"{e}: Failed to create file '{str(img_name)}'. File may exist or contain partial data")
                else:
                    relative_dataset_path = str(
                        Path(self.dataset_path).relative_to(DATA_DIR.parent))
                    log_info(
                        f"Successfully stored '{str(img_name)}' in \'{relative_dataset_path}\' ")
            return True

    def calc_total_filesize(self):
        if self.dataset:
            self.dataset_total_filesize = 0
            for data in self.dataset:
                self.dataset_total_filesize += data.size
            # To get size in MB
            self.dataset_total_filesize = bytes_divisor(
                self.dataset_total_filesize, -2)
        else:
            self.dataset_total_filesize = 0
        return self.dataset_total_filesize

    def save_dataset(self) -> bool:
        if self.dataset_path:
            directory_name = self.dataset_path
        else:  # if somehow dataset path Did Not Exist
            directory_name = get_directory_name(
                self.name)  # change name to lowercase
            # join directory name with '-' dash
            self.dataset_path = DATA_DIR / 'dataset' / str(directory_name)

        # directory_name = get_directory_name(
        #     self.name)
        # self.dataset_path = Path.home() / '.local' / 'share' / \
        #     'integrated-vision-inspection-system' / \
        #     'app_media' / 'dataset' / str(directory_name)
        # self.dataset_path = Path(self.dataset_path)

        create_folder_if_not_exist(self.dataset_path)
        if self.dataset_PNG_encoding():
            # st.success(f"Successfully created **{self.name}** dataset")
            return self.dataset_path


class NewDataset(BaseDataset):
    def __init__(self, dataset_id) -> None:
        # init BaseDataset -> Temporary dataset ID from random gen
        super().__init__(dataset_id)
        self.dataset_total_filesize = 0  # in byte-size

    # removed deployment type and insert filetype_id select from public.filetype table

    def insert_dataset(self):
        insert_dataset_SQL = """
                                INSERT INTO public.dataset (
                                    name,
                                    description,
                                    dataset_path,
                                    dataset_size,
                                    filetype_id)
                                VALUES (
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    
                                    (SELECT ft.id from public.filetype ft where ft.name = %s))
                                RETURNING id;
                            """
        insert_dataset_vars = [self.name, self.desc,
                               str(self.dataset_path), self.dataset_size, self.filetype]
        self.dataset_id = db_fetchone(
            insert_dataset_SQL, conn, insert_dataset_vars).id
        return self.dataset_id


class Dataset(BaseDataset):
    def __init__(self, dataset, data_name_list: Dict = {}) -> None:
        super(). __init__(dataset.ID)
        self.id = dataset.ID
        self.name = dataset.Name
        self.desc = dataset.Description
        self.dataset_size = dataset.Dataset_Size
        self.dataset_path = dataset.Dataset_Path
        self.filetype = dataset.File_Type
        self.data_name_list = {}

    def get_data_name_list(self, data_name_list_full: Dict):
        """Obtain list of data in the dataset 
            - Iterative glob through the dataset directory
            - Obtain filename using pathlib.Path(<'filepath/*'>).name

        Returns:
            Dict[dict]: Dataset name as key to a List of data in the dataset directory
        """
        try:
            # IF dataset info already exist and len of data same as number of files in folder -> get from Dict
            if data_name_list_full.get(self.name) and (len(data_name_list_full.get(self.name))) == len([file for file in Path(self.dataset_path).iterdir() if file.is_file()]):

                self.data_name_list = data_name_list_full.get(self.name)

            else:
                data_name_list_full = self.glob_folder_data_list(
                    data_name_list_full)

        except AttributeError as e:
            log_error(f"{e}: NoneType error for data_name_list dictionary")
            data_name_list_full = {}

        return data_name_list_full

    def glob_folder_data_list(self, data_name_list_full: Dict):
        if self.dataset_path:

            dataset_path = self.dataset_path + "/*"

            data_info_tmp = []

            # i need
            # {'id':data_name,'filetype':self.filetype,'created_at':os.stat().st_mtime}

            # Glob through dataset directory
            for data_path in iglob(dataset_path):
                data_info = {}

                log_info(f"Globbing {data_path}......")
                data_info['id'] = Path(data_path).name
                data_info['filetype'] = self.filetype

                # Get File Modified Time
                data_modified_time_epoch = os.stat(str(data_path)).st_mtime
                data_modified_time = datetime.fromtimestamp(data_modified_time_epoch
                                                            ).strftime('%Y-%m-%d')
                data_info['created'] = data_modified_time
                data_info_tmp.append(data_info)

            data_name_list_full[self.name] = data_info_tmp
            self.data_name_list = data_info_tmp

            return data_name_list_full

    def update_dataset_size(self):
        new_dataset_size = len([file for file in Path(
            self.dataset_path).iterdir() if file.is_file()])

        update_dataset_size_SQL = """
                                    UPDATE
                                        public.dataset
                                    SET
                                        dataset_size = %s
                                    WHERE
                                        id = %s
                                    RETURNING dataset_size;
        
                                    """
        update_dataset_size_vars = [new_dataset_size, self.id]
        new_dataset_size_return = db_fetchone(
            update_dataset_size_SQL, conn, update_dataset_size_vars)
        self.dataset_size = new_dataset_size_return if new_dataset_size_return else self.dataset_size

    def update_title_desc(self, new_name: str, new_desc: str):
        update_title_desc_SQL = """
                                    UPDATE
                                        public.dataset
                                    SET
                                        name = %s,
                                        description = %s
                                    WHERE
                                        id = %s
                                    RETURNING name,description;
        
                                    """
        update_title_desc_vars = [new_name, new_desc, self.id]
        new_title_desc_return = db_fetchone(
            update_title_desc_SQL, conn, update_title_desc_vars)
        log_info(f"Updating title and desc {new_title_desc_return}")
        sleep(1)


        self.name, self.desc = new_title_desc_return if new_title_desc_return else (
            None, None)

    def update_dataset(self):
        """Wrapper function to update existing dataset
        """
        # save added data into file-directory
        return self.save_dataset()

    def update_pipeline(self, success_place) -> int:
        """ Pipeline to update dataset

        Args:
            success_place (EmptyMixin): Placeholder to display dataset update progress and info

        Returns:
            int: Return append_data_flag as 0 to leave *data_upload_module*
        """

        if self.update_dataset():

            success_place.success(
                f"Successfully appended **{self.name}** dataset")

            sleep(1)

            success_place.empty()

            self.update_dataset_size()

            log_info(
                f"Successfully updated **{self.name}** size in database")

            append_data_flag = 0

        else:
            success_place.error(
                f"Failed to append **{self.name}** dataset")
            append_data_flag = 1

        return append_data_flag


# TODO REMOVE


    def update_data_name_list(self, new_name_list: List):
        current_data_name_only_list = [x['id'] for x in self.data_name_list]
        new_data_name_only_list = list(
            set(current_data_name_only_list + new_name_list))
        # NOT NEEDED
        # Image Previewer


# TODO Observe query dataset list caching performance
# @st.cache(ttl=100)


def query_dataset_list() -> List[namedtuple]:
    """Query list of dataset from DB, Column Names: TRUE

    Returns:
        namedtuple: List of datasets
    """
    query_dataset_SQL = """
        SELECT
            id AS "ID",
            name AS "Name",
            dataset_size AS "Dataset Size",
            (SELECT ft.name AS "File Type" from public.filetype ft where ft.id = d.filetype_id),
            updated_at AS "Date/Time",
            description AS "Description",
            dataset_path AS "Dataset Path"
        FROM
            public.dataset d
        ORDER BY id ASC;"""

    datasets, column_names = db_fetchall(
        query_dataset_SQL, conn, fetch_col_name=True)
    log_info("Querying dataset from database......")
    dataset_tmp = []
    if datasets:
        for dataset in datasets:

            # convert datetime with TZ to (2021-07-30 12:12:12) format
            converted_datetime = dataset.Date_Time.strftime(
                '%Y-%m-%d %H:%M:%S')

            dataset = dataset._replace(
                Date_Time=converted_datetime)
            dataset_tmp.append(dataset)

        # self.dataset_list = dataset_tmp
    else:
        dataset_tmp = []

    return dataset_tmp, column_names



def get_dataset_name_list(dataset_list: List[namedtuple]):
    """Generate Dictionary of namedtuple

    Args:
        dataset_list (List[namedtuple]): Query from database

    Returns:
        Dict: Dictionary of namedtuple
    """

    # dataset_name_list = {}  # list of dataset name for selectbox
    dataset_dict = {}  # use to store named tuples as value to dataset name as key

    if dataset_list:
        for dataset in dataset_list:
            # DEPRECATED -> dataset info can be accessed through namedtuple of dataset_dict
            # dataset_name_list[dataset.Name] = dataset.ID
            dataset_dict[dataset.Name] = dataset
        log_info("Generating list of dataset names and ID......")
    return dataset_dict


def main():
    print("Hi")


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
