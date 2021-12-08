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
from core.utils.log import logger  # logger
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


def check_if_field_empty(context: Dict, field_placeholder, name_key: str = 'name', check_if_exists=None):
    empty_fields = []

    # if not all_field_filled:  # IF there are blank fields, iterate and produce error message
    for k, v in context.items():
        if v and v != "":
            # also check whether the specific name exists in the database
            # or if the name is too long to be used for directory name
            if (k == name_key):
                # after some testing, 21 should be the maximum length
                # to avoid reaching the 255 (or 260?) limit for Windows path
                if len(v) > 21:
                    logger.error("Name should be less than 21 characters long, "
                                 "please use a shorter name")
                    field_placeholder[k].error(
                        "Name should be less than 21 characters long, "
                        "please use a shorter name")
                    empty_fields.append(k)
                    continue

                context = {'column_name': 'name', 'value': v}

                if check_if_exists(context, conn):
                    field_placeholder[k].error(
                        f"Project name used. Please enter a new name")
                    sleep(1)
                    field_placeholder[k].empty()
                    logger.error(
                        f" name used. Please enter a new name")
                    empty_fields.append(k)
                else:
                    logger.debug('escaped check')
        else:
            field_placeholder[k].error(
                f"Please do not leave field blank")
            empty_fields.append(k)
    logger.debug(f"{empty_fields = }")
    # if empty_fields not empty -> return True, else -> return False (Negative Logic)
    return not empty_fields  # Negative logic


def remove_newline_trailing_whitespace(text: str) -> str:
    fixed_text = " ".join(text.split())
    return fixed_text


def reset_page_attributes(attributes_list: List):
    for attrib in attributes_list:
        if attrib in session_state:
            logger.debug(f"del {attrib}")
            del session_state[attrib]
