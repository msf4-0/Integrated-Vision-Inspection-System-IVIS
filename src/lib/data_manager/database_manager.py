"""
Title: Database
Date: 26/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

# Initialise Connection Snippet
import sys
import psycopg2
import logging
import streamlit as st
# from config import config

#--------------------Logger-------------------------#

FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
DATEFMT = '%d-%b-%y %H:%M:%S'

# logging.basicConfig(filename='test.log',filemode='w',format=FORMAT, level=logging.DEBUG)
logging.basicConfig(format=FORMAT, level=logging.INFO,
                    stream=sys.stdout, datefmt=DATEFMT)

log = logging.getLogger()

#----------------------------------------------------#

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
        log.info('Connecting to the PostgreSQL database...')
        if kwargs:
            conn = psycopg2.connect(**kwargs)
        else:
            conn = psycopg2.connect(dsn, connection_factory, cursor_factory)

        # create a cursor
        cur = conn.cursor()

        # execute a statement
        log.info('PostgreSQL database version:')
        cur.execute('SELECT version()')

        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        log.info(db_version)

        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        log.info(error)
        conn = None
    # finally:
    #     if conn is not None:
    #         conn.close()
    #         print('Database connection closed.')
    return conn
