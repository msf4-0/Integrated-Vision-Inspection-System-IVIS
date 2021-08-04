"""
Title: Helper
Date: 19/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
from typing import Union, List, Dict, Optional
from time import perf_counter
import pandas as pd
import psycopg2
from psycopg2 import sql
import mimetypes
from streamlit.uploaded_file_manager import UploadedFile
import streamlit as st
# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[3]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
from data_manager.database_manager import db_fetchone, init_connection

conn = init_connection(**st.secrets["postgres"])


def split_string(string: str) -> List:

    # Split the string based on space delimiter
    list_string = string.split(' ')

    return list_string


def join_string(list_string: List) -> str:

    # Join the string based on '-' delimiter
    string = '-'.join(list_string)

    return string


def get_directory_name(name: str) -> str:
    directory_name = join_string(split_string(str(name))).lower()
    return directory_name


def is_empty(iterable: Union[List, Dict, set]) -> bool:
    return not bool(iterable)


@st.cache
def create_dataframe(data: Union[List, Dict, pd.Series], column_names: List = None, sort: bool = False, sort_by: Optional[str] = None, asc: bool = True, date_time_format: bool = False) -> pd.DataFrame:
    if data:

        df = pd.DataFrame(data, columns=column_names)
        df.index.name = ('No.')
        if date_time_format:
            df['Date/Time'] = pd.to_datetime(df['Date/Time'],
                                             format='%Y-%m-%d %H:%M:%S')

            # df.sort_values(by=['Date/Time'], inplace=True,
            #                ascending=False, ignore_index=True)
        if sort:

            df.sort_values(by=[sort_by], inplace=True,
                           ascending=asc, ignore_index=True)

            # dfStyler = df.style.set_properties(**{'text-align': 'center'})
            # dfStyler.set_table_styles(
            #     [dict(selector='th', props=[('text-align', 'center')])])

        return df


def check_if_exists(table: str, column_name: str, condition, conn):
    # Separate schema and tablename from 'table'
    schema, tablename = [i for i in table.split('.')]
    check_if_exists_SQL = sql.SQL("""
                        SELECT
                            EXISTS (
                                SELECT
                                    *
                                FROM
                                    {}
                                WHERE
                                    {} = %s);
                            """).format(sql.Identifier(schema, tablename), sql.Identifier(column_name))
    check_if_exists_vars = [condition]
    exist_flag = db_fetchone(check_if_exists_SQL, conn, check_if_exists_vars)

    return exist_flag

# MIME: type/subtype
# get filetype


def get_filetype(file: Union[str, Path, UploadedFile]):
    """Get filetype from MIME of the file <type/subtype>
        Eg. image,video,audio,text

    Args:
        file (Union[str,Path,UploadedFile]): File can be string path or Path-like object or Streamlit's UploadedFile object

    Returns:
        string: filetype
    """
    if isinstance(file, (str, Path)):
        mime_type = mimetypes.guess_type(file)[0]
        # filetype = str(Path(mime_type).parent)
    elif isinstance(file, UploadedFile):
        log_info(f"File: {file}")
        mime_type = file.type
    if mime_type:
        filetype = str(Path(mime_type).parent)
    else:
        filetype = None
    return filetype


def compare_filetypes(file_tuple: tuple):
    """Compare if 2 instances are equal, else break loop 

    Args:
        file_tuple (tuple): A pair of elements to be compared -> 2 elements zip of a List or Dict

    Returns:
        bool: True is if equal, else False
    """
    filetype1, filetype2 = list(map(get_filetype, file_tuple))  # updated
    log_info(f"File tuple:{file_tuple}")
    if filetype1 == filetype2:
        log_info(f"File types:{filetype1,filetype2}")
        return True, filetype1, file_tuple[0].name, file_tuple[1].name
    else:

        return False, filetype1, file_tuple[0].name, file_tuple[1].name


def check_filetype(uploaded_files, dataset, field_placeholder: Dict = None):
    """Constraint for only one type of files (Image, Video, Audio, Text)

    1. Image: .jpg, .png, .jpeg
    2. Video: .mp4, .mpeg
    3. Audio: .wav, .mp3, .m4a
    4. Text: .txt, .csv

    Args:
        uploaded_files (Union[str,Path,UploadedFile], optional): [description]. Defaults to session_state.upload_widget.
    """
    uploaded_files: Union[str, Path, UploadedFile] = uploaded_files
    if uploaded_files:
        start_time = perf_counter()
        if len(uploaded_files) == 1:
            filetype = get_filetype(uploaded_files[0])
            log_info("Enter single")
            log_info(filetype)

        else:
            log_info("Enter multi")
            filetypes = map(compare_filetypes, zip(
                uploaded_files[:], uploaded_files[1:]))
            for check_result, filetype, file1, file2 in filetypes:
                if check_result:
                    log_info("Filetype passed")
                    pass
                else:
                    filetype_error_msg = f"Filetype different for {file1} and {file2}"
                    log_error(filetype_error_msg)
                    if field_placeholder:
                        field_placeholder["upload"].error(
                            filetype_error_msg)
                    break

                    # GET Filetype
        dataset.filetype = filetype.capitalize()

        end_time = perf_counter()
        time_elapsed = end_time - start_time
        number_of_files = len(uploaded_files)
        average_time = time_elapsed / number_of_files
        log_info(
            f"Time taken to compare filetypes {time_elapsed}s with average of {average_time}s for {number_of_files}")
