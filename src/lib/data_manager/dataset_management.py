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
from typing import Dict, Union, List
from enum import IntEnum
from PIL import Image
from time import sleep
from glob import iglob
from datetime import datetime
from stqdm import stqdm
from videoprops import get_audio_properties, get_video_properties
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
from path_desc import chdir_root, MEDIA_ROOT, BASE_DATA_DIR, DATASET_DIR
from core.utils.log import log_info, log_error  # logger
from data_manager.database_manager import init_connection, db_fetchone, db_fetchall
from core.utils.file_handler import bytes_divisor, create_folder_if_not_exist
from core.utils.helper import get_directory_name, get_mime
from core.utils.form_manager import check_if_exists, check_if_field_empty, reset_page_attributes
from core.utils.dataset_handler import get_image_size

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])
# >>>> Variable Declaration <<<<


class DataPermission(IntEnum):
    ViewOnly = 0
    Edit = 1

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return DataPermission[s]
        except KeyError:
            raise ValueError()


class DatasetPagination(IntEnum):
    Dashboard = 0
    New = 1

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return DatasetPagination[s]
        except KeyError:
            raise ValueError()


class FileTypes(IntEnum):
    Image = 0
    Video = 1
    Audio = 2
    Text = 3

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return FileTypes[s]
        except KeyError:
            raise ValueError()

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
        self.raw_data_dict: Dict = {}  # stores raw data of data

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

    def check_if_field_empty(self, context: Dict, field_placeholder):
        check_if_exists = self.check_if_exists
        empty_fields = check_if_field_empty(
            context, field_placeholder, check_if_exists)
        return empty_fields

    def check_if_exists(self, context: Dict, conn) -> bool:
        table = 'public.dataset'

        exists_flag = check_if_exists(
            table, context['column_name'], context['value'], conn)

        return exists_flag

    def dataset_PNG_encoding(self):
        if self.dataset:
            for img in stqdm(self.dataset, unit=self.filetype, ascii='123456789#', st_container=st.sidebar, desc="Uploading data"):
                img_name = img.name
                log_info(img.name)
                save_path = Path(self.dataset_path) / str(img_name)

                # st.title(img.name)
                try:
                    with Image.open(img) as pil_img:
                        pil_img.save(save_path)
                except ValueError as e:
                    log_error(
                        f"{e}: Could not resolve output format for '{str(img_name)}'")
                except OSError as e:
                    log_error(
                        f"{e}: Failed to create file '{str(img_name)}'. File may exist or contain partial data")
                else:
                    relative_dataset_path = str(
                        Path(self.dataset_path).relative_to(BASE_DATA_DIR))
                    log_info(
                        f"Successfully stored '{str(img_name)}' in \'{relative_dataset_path}\' ")
            return True

    def calc_total_filesize(self):
        if self.dataset:
            self.dataset_total_filesize = 0
            for data in self.dataset:
                self.dataset_total_filesize += data.size
            # # To get size in MB
            # self.dataset_total_filesize = bytes_divisor(
            #     self.dataset_total_filesize, -2)
        else:
            self.dataset_total_filesize = 0
        return self.dataset_total_filesize

    @staticmethod
    def get_dataset_path(dataset_name: str) -> Path:
        directory_name = get_directory_name(
            dataset_name)  # change name to lowercase
        # join directory name with '-' dash
        dataset_path = DATASET_DIR / str(directory_name)
        log_info(f"Dataset Path: {dataset_path}")
        return dataset_path

    def save_dataset(self) -> bool:

        # Get absolute dataset folder path
        self.dataset_path = self.get_dataset_path(self.name)

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
                                    dataset_size,
                                    filetype_id)
                                VALUES (
                                    %s,
                                    %s,
                                    %s,
                                    
                                    (SELECT ft.id from public.filetype ft where ft.name = %s))
                                RETURNING id;
                            """
        insert_dataset_vars = [self.name, self.desc,
                               self.dataset_size, self.filetype]
        self.dataset_id = db_fetchone(
            insert_dataset_SQL, conn, insert_dataset_vars).id
        return self.dataset_id

    @staticmethod
    def reset_new_dataset_page():
        """Method to reset all widgets and attributes in the New Dataset Page when changing pages
        """

        new_dataset_attributes = ["new_dataset", "data_source_radio", "name",
                                  "desc", "upload_widget"]

        reset_page_attributes(new_dataset_attributes)


class Dataset(BaseDataset):
    def __init__(self, dataset, data_name_list: Dict = {}) -> None:
        super(). __init__(dataset.ID)
        self.id = dataset.ID
        self.name = dataset.Name
        self.desc = dataset.Description
        self.dataset_size = dataset.Dataset_Size
        self.dataset_path = self.get_dataset_path(self.name)  # NOTE
        # self.dataset_path = dataset.Dataset_Path
        self.filetype = dataset.File_Type
        self.data_name_list = {}

    def get_data_name_list(self, data_name_list_full: Dict):
        """Obtain list of data in the dataset 
            - Iterative glob through the dataset directory
            - Obtain filename using pathlib.Path(<'filepath/*'>).name

        Returns:
            Dict[dict]: Dataset name as key to a List of data in the dataset directory
        """
        self.dataset_path = self.get_dataset_path(self.name)
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
        self.dataset_path = self.get_dataset_path(self.name)
        log_info(self.dataset_path)
        log_info(self.name)
        if self.dataset_path:

            dataset_path = Path(self.dataset_path) / "./*"

            data_info_tmp = []

            # i need
            # {'id':data_name,'filetype':self.filetype,'created_at':os.stat().st_mtime}

            # Glob through dataset directory
            for data_path in iglob(str(dataset_path)):
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
        self.dataset_path = self.get_dataset_path(self.name)
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

        self.name, self.desc = new_title_desc_return if new_title_desc_return else (
            None, None)

    def update_dataset_path(self, new_dataset_name: str):
        # new_directory_name = get_directory_name(new_dataset_name)
        # new_dataset_path = DATASET_DIR / str(new_directory_name)
        new_dataset_path = self.get_dataset_path(new_dataset_name)

        # Rename dataset folder
        try:
            old_dataset_path = Path(self.dataset_path)
            old_dataset_path.rename(Path(new_dataset_path))
            log_info(f'Renamed dataset path to {new_dataset_path}')
        except Exception as e:
            log_error(f'{e}: Could not rename file')

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

    def display_data_media_attributes(self, data_info: str, data_raw: Image.Image, filename: str = None, placeholder=None):
        if placeholder:
            placeholder = placeholder
        else:
            placeholder = st.empty()

        if data_info:
            data_name = data_info['id'] if data_info['id'] else Path(
                data_raw.filename).name
            created = data_info['created'] if data_info['created'] else ""
            mimetype = get_mime(data_name)

            if not data_raw:
                self.dataset_path = self.get_dataset_path(self.name)
                filepath = self.dataset_path / data_name

            try:
                filetype = data_info['filetype']
            except:
                filetype = str(Path(mimetype).parent)

            if isinstance(filetype, str):
                filetype = FileTypes.from_string(filetype)
            # Image
            if filetype == FileTypes.Image:
                image_width, image_height = get_image_size(data_raw)
                display_attributes = f"""
                ### **DATA**
                - Name: {data_name}
                - Created: {created}
                - Dataset: {self.name}
                ___
                ### **MEDIA ATTRIBUTES**
                - Width: {image_width}
                - Height: {image_height}
                - MIME type: {mimetype}
                """
            # video
            elif filetype == FileTypes.Video:
                props = get_video_properties(filepath)

                display_attributes = f"""
                ### **DATA**
                - Name: {data_name}
                - Created: {created}
                - Dataset: {self.name}
                ___
                ### **MEDIA ATTRIBUTES**
                - Codec:{props['codec_name']}
                - Width: {props['width']}
                - Height: {props['height']}
                - Duration: {float(props['duration']):.2f}s
                - Frame rate: {props['avg_frame_rate']}
                - Frame count: {props['nb_frames']}
                - MIME type: {mimetype}
                """
            # Audio
            elif filetype == FileTypes.Audio:

                props = get_audio_properties(filepath)

                display_attributes = f"""
                ### **DATA**
                - Name: {data_name}
                - Created: {created}
                - Dataset: {self.name}
                ___
                ### **MEDIA ATTRIBUTES**
                - Codec:{props['codec_long_name']}
                - Channel Layout: {props['channel_layout']}
                - Channels: {props['channels']}
                - Duration: {float(props['duration']):.2f}s
                - Bit rate: {(float(props['bit_rate'])/1000)}kbps
                - Sampling rate: {(float(props['sample_rate'])/1000):.2f}kHz
                - MIME type: {mimetype}
                """
            # Text
            elif filetype == FileTypes.Text:
                display_attributes = f"""
                ### **DATA**
                - Name: {data_name}
                - Created: {created}
                - Dataset: {self.name}
                ___
                ### **MEDIA ATTRIBUTES**
                - MIME type: {mimetype}
                """
            if placeholder:
                with placeholder.container():
                    st.info(display_attributes)
            else:
                st.write("### \n")
                st.info(display_attributes)

    def load_dataset(self):

        self.dataset_path = self.get_dataset_path(self.name)

        for file in Path(self.dataset_path).iterdir():
            if file.is_file():
                pass


# **************************** IMPORTANT ****************************
# ************************ CANNOT CACHE !!!!!*************************
# Will throw ValueError for selectbox dataset_sel because of session state (BUG)
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
            description AS "Description"
            
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

# **************************** IMPORTANT ****************************
# ************************ CANNOT CACHE !!!!!*************************
# Will throw ValueError for selectbox dataset_sel because of session state (BUG)


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
