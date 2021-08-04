"""
Title: Helper
Date: 19/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
from typing import Union, List, Dict, Optional
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
        filetype = str(Path(mime_type).parent)
    elif isinstance(file, UploadedFile):
        mime_type = file.type

    filetype = str(Path(mime_type).parent)
    return filetype


def compare_filetypes(file_tuple: tuple):
    """Compare if 2 instances are equal, else break loop 

    Args:
        file_tuple (tuple): A pair of elements to be compared -> 2 elements zip of a List or Dict

    Returns:
        bool: True is if equal, else False
    """
    filetype1, filetype2 = file_tuple
    print(file_tuple)
    if filetype1 == filetype2:
        return True
    else:

        return False
