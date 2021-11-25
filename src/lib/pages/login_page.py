"""
Title: Login Page
Date: 23/6/2021
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
from pathlib import Path
from time import sleep
import sys

import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state

# >>>>>>>>>>>>>>>>>>>>>PATH>>>>>>>>>>>>>>>>>>>>>
SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
TEST_MODULE_PATH = SRC / "test" / "test_page" / "module"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

if str(TEST_MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(TEST_MODULE_PATH))
# <<<<<<<<<<<<<<<<<<<<<<PATH<<<<<<<<<<<<<<<<<<<<<<<

# DEFINE Web APP page configuration
# layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>> User-defined modules >>>>
from path_desc import chdir_root
from core.utils.log import logger
from data_manager.database_manager import init_connection
from user.user_management import LoginPagination, User, UserLogin, AccountStatus, check_if_field_empty, reset_login_page
from main_page_management import MainPagination

# <<<< User-defined modules <<<<

# >>>> Variable declaration >>>>
user_test = {"username": 'chuzhenhao', "psd": "shrdc", "status": "NEW"}
conn = init_connection(**st.secrets["postgres"])


def activation_page(user: UserLogin = None, layout='wide'):  # activation page for new users
    if not user:
        if 'user_login' not in session_state:
            st.warning("No user to be activated.")
            st.stop()
        user = session_state.user_login

    psd_place = {}
    psd = {}
    FIELDS = {
        'first': 'New Password',
        'second': 'Confirm Password'
    }

    st.write("## __User Account Activation__")
    st.markdown("___")
    activation_place = st.empty()
    if layout == 'wide':
        _, col2, _ = activation_place.columns([1, 3, 1])
    else:
        col2 = activation_place

    with col2.form(key="activate", clear_on_submit=True):
        # psd 1
        psd["first"] = st.text_input(label="New Password", key="new_password",
                                     type="password", help="Enter new password")
        psd_place["first"] = st.empty()

        # psd confirm
        psd["second"] = st.text_input(label="Confirm Password", key="confirm_password",
                                      type="password", help="Confirm Password")
        psd_place["second"] = st.empty()

        submit_activation = st.form_submit_button(
            "Confirm", help="Submit desired password to activate user account")
        if submit_activation:

            has_submitted = check_if_field_empty(psd, psd_place, FIELDS)
            if has_submitted:
                if psd["first"] == psd["second"]:
                    user.update_psd(psd["first"])

                    # update the user status to ACTIVE after their first success login
                    user.update_status(AccountStatus.ACTIVE)

                    st.success("""Successfully activated account.
                                Returning to Login Page.
                                """)
                    reset_login_page()
                    sleep(2)
                    session_state.login_pagination = LoginPagination.Login
                    st.experimental_rerun()

                else:
                    st.error(
                        "Activation failed, passwords does not match. Please enter again.")

    def to_login_cb():
        # login_page(layout)
        session_state.main_pagination = LoginPagination.Login

    with col2:
        st.button("Back", key="btn_back_to_login", on_click=to_login_cb)


def login_page(layout='wide'):
    FIELDS = {
        'username': 'Username',
        'psd': "Password"
    }
    user = {}  # store user input
    login_field_place = {}
    login_place = st.empty()  # PLACEHOLDER to replace with error message

    # >>>> Place login container at the centre when layout == 'wide'
    if layout == 'wide':
        left, mid_login, right = login_place.columns([1, 3, 1])
    else:  # Place login container at the centre when layout =='centered'
        mid_login = login_place

    with mid_login.form(key="login", clear_on_submit=True):

        st.write("## Login")

        # >>>>>>>> INPUT >>>>>>>>
        # 1. USERNAME
        user["username"] = st.text_input(
            "Username", key="username", help="Enter your username")
        login_field_place["username"] = st.empty()
        # 2. PASSWORD
        user["psd"] = st.text_input("Password", value="",
                                    key="safe", type="password")
        login_field_place["psd"] = st.empty()

        submit_login = st.form_submit_button("Log In")

       # st.write(f"{username},{pswrd}")
       # >>>>>>>> INPUT >>>>>>>>

        # user_login = UserLogin()  # Instatiate Temp login user
        success_place = st.empty()  # Placeholder for Login success

        # >>>>>>>> CHECK FIELD EMPTY >>>>>>>>
        if submit_login:  # when submit button is pressed
            has_submitted = check_if_field_empty(
                user, login_field_place, FIELDS)
        # <<<<<<<< CHECK FIELD EMPTY <<<<<<<<

            # >>>>>>>> VERIFICATION >>>>>>>>
            if has_submitted:  # if both fields entered

                if "user_login" not in session_state:
                    # Instantiate UserManager class SS holder
                    session_state.user_login = UserLogin()

                if session_state.user_login.user_verification(user, conn):

                    # >>>> CHECK user status >>>>
                    if session_state.user_login.status == AccountStatus.NEW:
                        session_state.login_pagination = LoginPagination.Activation
                        st.success("This is a new account, entering activation page "
                                   "to activate it.")
                        logger.info("This is a new account, entering activation page "
                                    "to activate it.")
                        st.experimental_rerun()
                    elif session_state.user_login.status == AccountStatus.LOCKED:

                        # admin_email = 'admin@shrdc.com'  # Random admin email
                        st.error(
                            f"Account Locked. Please contact any Administrator.")

                    # >>>>>>>> SUCCESS ENTER >>>>>>>>
                    else:
                        # for other status, enter web app
                        # set status as log-in
                        session_state.user_login.update_status(
                            AccountStatus.LOGGED_IN)

                        # Save Session Log
                        session_state.user_login.save_session_log()

                        success_place.success(
                            "#### You have logged in successfully. Welcome 👋🏻")
                        # change to User
                        session_state.user = User.from_user_login(
                            session_state.user_login)

                        sleep(2)
                        success_place.empty()

                        reset_login_page()
                        session_state.main_pagination = MainPagination.Projects
                        st.experimental_rerun()
                    # <<<< CHECK user status <<<<

                else:
                    st.error(
                        "User entered wrong username or password. Please enter again.")

            # <<<<<<<< VERIFICATION <<<<<<<<


def index():
    chdir_root()  # change to root directory

    LOGIN_PAGES = {
        LoginPagination.Login: login_page,
        LoginPagination.Activation: activation_page
    }

    if 'login_pagination' not in session_state:
        session_state.login_pagination = LoginPagination.Login

    logger.debug(f"Navigator: {session_state.login_pagination = }")
    LOGIN_PAGES[session_state.login_pagination]()


if __name__ == "__login__":
    if st._is_running_with_streamlit:
        index()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
