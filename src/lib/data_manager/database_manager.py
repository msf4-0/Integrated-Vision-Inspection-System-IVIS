"""
Title: Database
Date: 26/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

# Initialise Connection Snippet
import sys
from pathlib import Path
import psycopg2
import logging
import streamlit as st
# from config import config
# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
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


# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

dsn = "host=localhost port=5432 dbname=eye user=shrdc password=shrdc"

# Initialise Connection to PostgreSQL Database Server

# TODO: HANDLE KWARGS


@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
def init_connection(dsn=None, connection_factory=None, cursor_factory=None, **kwargs):
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        # params = config()

        # connect to the PostgreSQL server
        log_info('Connecting to the PostgreSQL database...')
        if kwargs:
            conn = psycopg2.connect(**kwargs)
        else:
            conn = psycopg2.connect(dsn, connection_factory, cursor_factory)

        # create a cursor
        cur = conn.cursor()

        # execute a statement
        log_info('PostgreSQL database version:')
        cur.execute('SELECT version()')

        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        log_info(db_version)

        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        log_error(error)
        conn = None
    # finally:
    #     if conn is not None:
    #         conn.close()
    #         print('Database connection closed.')
    return conn


def db_uni_query(sql_message, conn):
    with conn:
        with conn.cursor() as cur:
            try:
                cur.execute(sql_message)
                conn.commit()
            except psycopg2.Error as e:
                log_error(e)


def db_fetchone(sql_message, conn):
    with conn:
        with conn.cursor() as cur:
            try:
                cur.execute(sql_message)
                conn.commit()
                return_one = cur.fetchone()  # return tuple
            except psycopg2.Error as e:
                log_error(e)

    return return_one


def db_fetchall(sql_message, conn):
    with conn:
        with conn.cursor() as cur:
            try:
                cur.execute(sql_message)
                conn.commit()
                return_all = cur.fetchall()  # return array of tuple
            except psycopg2.Error as e:
                log_error(e)

    return return_all
