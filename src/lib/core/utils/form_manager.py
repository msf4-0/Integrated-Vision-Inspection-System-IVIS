import sys
from pathlib import Path
from typing import Dict
from psycopg2 import sql
from time import sleep
from typing import List
import streamlit as st
from streamlit import session_state as session_state

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


def check_if_exists(table: str, column_name: str, condition, conn=conn):

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
    exist_flag = db_fetchone(check_if_exists_SQL, conn,
                             check_if_exists_vars).exists

    return exist_flag


def check_if_field_empty(context: Dict, field_placeholder, name_key:str='name',check_if_exists=None):
    empty_fields = []

    # if not all_field_filled:  # IF there are blank fields, iterate and produce error message
    for k, v in context.items():
        if v and v != "":
            if (k == name_key):
                context = {'column_name': 'name', 'value': v}

                if check_if_exists(context, conn):
                    field_placeholder[k].error(
                        f"Project name used. Please enter a new name")
                    sleep(1)
                    field_placeholder[k].empty()
                    log_error(
                        f" name used. Please enter a new name")
                    empty_fields.append(k)
                else:
                    log_error('escaped check')

            else:

                pass
        else:

            field_placeholder[k].error(
                f"Please do not leave field blank")
            empty_fields.append(k)
    log_info(empty_fields)
    # if empty_fields not empty -> return True, else -> return False (Negative Logic)
    return not empty_fields  # Negative logic


def remove_newline_trailing_whitespace(text: str) -> str:
    fixed_text = " ".join([x for x in text.split()])

    return fixed_text


def reset_page_attributes(attributes_list: List):
    for attrib in attributes_list:
        if attrib in session_state:
            log_info(f"del {attrib}")
            del session_state[attrib]
