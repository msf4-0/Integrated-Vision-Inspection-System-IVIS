"""
Title: Database Setup
Date: 13/9/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)

Copyright (C) 2021 Selangor Human Resource Development Centre

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. 

Copyright (C) 2021 Selangor Human Resource Development Centre
SPDX-License-Identifier: Apache-2.0
========================================================================================
"""

import os
import sys
from pathlib import Path
from time import sleep
from typing import Dict

import streamlit as st
import toml
from streamlit import cli as stcli
from streamlit import session_state


# ***************** Add src/lib to path ***************************
SRC = Path(__file__).resolve().parent  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
# ***************** Add src/lib to path ***************************


from core.utils.log import logger
from data_manager.database_manager import APP_DATABASE_NAME, init_connection, initialise_database_pipeline, test_db_conn, DatabaseStatus
from path_desc import SECRETS_PATH

# initialise connection to Database
# conn = init_connection(**st.secrets["postgres"])
place = {}
# st.write(SECRETS_PATH)

# **********************************SESSION STATE ******************************

# Flag to set True if successful connection
if 'db_connect_flag' not in session_state:
    session_state.db_connect_flag = False

if 'database_status' not in session_state:
    session_state.database_status = DatabaseStatus.NotExist
# **********************************SESSION STATE ******************************


def check_if_field_empty(context: Dict) -> bool:
    """Check if all fields in the form are filled

    Args:
        context (Dict): Dictionary to store the required details to establish connection with the PostgreSQL server
        - host: Name of server host. eg. 'localhost'
        - port: Port number of the Database server. eg. 5432 (Default port number for PostgreSQL)
        - dbname: Name of database
        - user: Name of USER / ROLE
        - password: Password to the Database

    Returns:
        bool: True if all filled, otherwise False
    """
    empty_fields = []

    # if not all_field_filled:  # IF there are blank fields, iterate and produce error message
    for k, v in context.items():

        if not v and v == "":
            empty_fields.append(k)
    logger.debug(empty_fields)

    # if empty_fields not empty -> return True, else -> return False (Negative Logic)
    return not empty_fields  # Negative logic


def test_database_connection(**dsn: Dict):
    """Test connection with the PostgreSQL database

    Returns:
        psycopg2.connection: Connection object for the Database
    """
    conn = test_db_conn(**dsn)
    if conn is not None:

        success_msg = f"Successfully connected to Database {dsn['dbname']}"
        logger.info(success_msg)
        success_place = st.empty()
        success_place.success(success_msg)
        sleep(0.7)
        success_place.empty()
    else:
        st.error(f"Failed to connect to the Database")
        logger.error(f"Failed to connect to the Database")
    return conn


def modify_secrets_toml(**context: Dict):
    if test_database_connection(**context):
        # connect to DB without using streamlit cache
        conn = test_db_conn(**context)
        database_status = initialise_database_pipeline(
            conn, context)

        if database_status == DatabaseStatus.Exist:
            # Write to secrets.toml file if database configuration is valid
            context['dbname'] = APP_DATABASE_NAME
            secrets = {'postgres': context}
            with open(str(SECRETS_PATH), 'w+') as f:
                new_toml = toml.dump(secrets, f)
                logger.info(f"Created secrets.toml file:\n{new_toml}")
                logger.debug(f"{SECRETS_PATH = }")
                logger.debug(f"{SECRETS_PATH.exists() = }")
            return

    logger.error("There were some error creating the database config or "
                 "connecting to database")


def db_config_form():
    """Database Configuration Form
    """
    def create_secrets_toml():
        """Create secrets.toml file with fields entered into the Form
        """
        context = {
            'host': session_state.host,
            'port': session_state.port,
            'dbname': session_state.dbname,
            'user': session_state.user_input,
            'password': session_state.password_input
        }

        if check_if_field_empty(context):

            # Modify secrets.toml file, and create a new database if not exists,
            #  based on the given db name
            modify_secrets_toml(**context)

            st.experimental_rerun()
        else:
            error_place = st.empty()
            error_place.error("Please fill in all fields")
            sleep(0.7)
            error_place.empty()
    st.title("Creating Database Configuration")
    st.markdown("___")
    st.info(f"""
    #### Please make sure the User has Create DB permission
    """)
    with st.form(key='db_setup', clear_on_submit=False):

        # Host
        st.text_input(label='Host', value='localhost', key='host',
                      help="This must be `localhost` if installing our application on a local machine.")

        # Port
        st.text_input(label='Port', value='5432', key='port',
                      help="By default, PostgreSQL database uses port 5432. "
                      "Please do not change this if you are not sure about it.")

        # dbname
        st.text_input(
            label='Existing Database Name', value='postgres',
            key='dbname', help="""This **postgres** database should exist by default when
            you installed PostgreSQL, unless you have deleted this default database,
            for which you will have to specify another database name in order for our
            application to connect to, as this is required to create a new database for
            our application. """)
        st.markdown("""Our new database will be named as **integrated_vision_inspection_system**.
            If this database already exists in your system, we will just assume that it's 
            the same database that we have created previously, and directly use it to query
            and also store the data for our application.""")

        st.markdown("**Notes about user and password:**")
        st.markdown("""This should be based on the user and password used during 
            your PostgreSQL installation.""")
        # user
        # session_state.user will conflict with our logged in user's session_state
        # so must use another key besides 'user'
        st.text_input(
            label='User', value='postgres', key='user_input',
            help="The default admin user should be **postgres**, change this if it is not.")

        # password
        st.text_input(label='Password',
                      key='password_input', type='password')

        if st.form_submit_button(label='Submit'):
            create_secrets_toml()


def database_setup():
    """Function to show Database Configuration Form if failed to connect DB
    - Check if secrets.toml file exists
    - If not, create a new TOML file at the default directory PROJECT_ROOT /.streamlit / secrets.toml

    """

    # If secretes.toml does not exists
    if not SECRETS_PATH.exists():

        # Ask user to enter Database details

        if not session_state.get('db_connect_flag'):

            db_config_form()

    else:
        # If connection to wrong database
        if st.secrets['postgres']['dbname'] != APP_DATABASE_NAME:

            if test_database_connection(**st.secrets['postgres']):
                session_state.db_connect_flag = True
                conn = test_db_conn(**st.secrets['postgres'])
                modify_secrets_toml(**st.secrets['postgres'])

            else:
                db_config_form()
        elif not session_state.get('db_connect_flag'):
            if test_database_connection(**st.secrets['postgres']):
                session_state.db_connect_flag = True
                conn = test_db_conn(**st.secrets['postgres'])
            else:
                db_config_form()


def database_direct_setup():
    host = "db" if os.environ.get("DOCKERCOMPOSE") else "localhost"
    db_config = {
        "host": host,
        # this port MUST be 5432 for PostgreSQL as default in Docker
        "port": '5432',
        # user should be 'postgres' as this is the default superuser
        # that has access to create database
        "user": 'postgres',
        # the rest are obtained from the environment variables
        # defined in docker-compose.yml
        "dbname": APP_DATABASE_NAME,
        "password": os.environ.get('POSTGRES_PASSWORD', 'shrdc')
    }
    logger.debug(f"{db_config = }")
    modify_secrets_toml(**db_config)


def main():
    pass


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        logger.info('[INFO] Setting up database using Streamlit ...')
        database_setup()
    elif os.environ.get('DOCKERCONTAINER'):
        logger.info('[INFO] Setting up database for Docker container ...')
        database_direct_setup()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
