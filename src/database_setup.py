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

from core.utils.file_handler import create_folder_if_not_exist
from core.utils.log import log_error, log_info  # logger
from data_manager.database_manager import db_no_fetch, init_connection
from path_desc import DATABASE_DIR, PROJECT_ROOT

# initialise connection to Database
# conn = init_connection(**st.secrets["postgres"])
place = {}
SECRETS_DIR = PROJECT_ROOT / ".streamlit/secrets.toml"
st.write(SECRETS_DIR)

# **********************************SESSION STATE ******************************
if 'db_connect_flag' not in session_state:
    session_state.db_connect_flag = False
# **********************************SESSION STATE ******************************

create_folder_if_not_exist(DATABASE_DIR)

def check_if_field_empty(context: Dict) -> bool:
    empty_fields = []

    # if not all_field_filled:  # IF there are blank fields, iterate and produce error message
    for k, v in context.items():

        if not v and v == "":

            # field_placeholder[k].error(
            #     f"Please do not leave field blank")
            empty_fields.append(k)
    log_info(empty_fields)
    # if empty_fields not empty -> return True, else -> return False (Negative Logic)
    return not empty_fields  # Negative logic


def test_database_connection(**dsn:Dict):
    conn = init_connection(**dsn)
    if conn != None:

        success_msg = f"Successfully connected to Database {dsn['dbname']}"
        log_info(success_msg)
        success_place=st.empty()
        success_place.success(success_msg)
        sleep(0.7)
        success_place.empty()
    else:
        st.error(f"Failed to connect to the Database")
    return conn


def db_input_form():
    def create_secrets_toml():
        context = {
            'host': session_state.host,
            'port': session_state.port,
            'dbname': session_state.dbname,
            'user': session_state.user,
            'password': session_state.password
        }
        if check_if_field_empty(context):

            secrets = {
                'postgres': context
            }
            if test_database_connection(**context):
                with open(str(SECRETS_DIR), 'w+') as f:
                    new_toml = toml.dump(secrets, f)
                    log_info(new_toml)
            
            st.experimental_rerun()
        else:
            error_place=st.empty()
            error_place.error("Please fill in all fields")
            error_place.empty()

    with st.form(key='db_setup', clear_on_submit=True):

        # Host
        st.text_input(label='Host', key='host')
        place['host'] = st.empty()
        # Port
        st.text_input(label='Port', key='port')
        place['port'] = st.empty()
        # dbname
        st.text_input(label='Database Name', key='dbname')
        place['dbname'] = st.empty()
        # user
        st.text_input(label='User', key='user')
        place['user'] = st.empty()
        # password
        st.text_input(label='Password',
                      key='password', type='password')
        place['password'] = st.empty()

        st.form_submit_button(label='Submit',
                              on_click=create_secrets_toml)
    


def database_setup():
    # st.write(st.secrets['postgres'])
    if not SECRETS_DIR.exists():
        # Ask user to enter Database details

        if not session_state.db_connect_flag:

            db_input_form()

    else:
        if not session_state.db_connect_flag:
            if test_database_connection(**st.secrets['postgres']):
                session_state.db_connect_flag = True
                conn = init_connection(**st.secrets['postgres'])
                return conn

            else:
                db_input_form()
            
    st.write(session_state.db_connect_flag)


def main():
    pass


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        database_setup()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
