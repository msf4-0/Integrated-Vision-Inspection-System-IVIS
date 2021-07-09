"""
Title: Login Page
Date: 23/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state as SessionState
from pathlib import Path
from time import sleep
import sys
import logging
import psycopg2

# >>>>>>>>>>>>>>>>>>>>>PATH>>>>>>>>>>>>>>>>>>>>>
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
# <<<<<<<<<<<<<<<<<<<<<<PATH<<<<<<<<<<<<<<<<<<<<<<<

# DEFINE Web APP page configuration
layout = 'wide'
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>> User-defined modules >>>>
from user_management import user_login, update_psd
from path_desc import chdir_root
from core.utils.log import std_log  # logger

# <<<< User-defined modules <<<<



@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
def init_connection():
    std_log(f"Connected to database {st.secrets['postgres']['dbname']} at PORT {st.secrets['postgres']['port']}")
    return psycopg2.connect(**st.secrets["postgres"])



# >>>> Variable declaration >>>>
user_test = {"username": 'chuzhenhao', "psd": "shrdc", "status": "NEW"}
user={}
login_field_place={}
conn=init_connection()
FIELDS={
    'username':'Username',
    'psd':"Password"
}


def check_if_field_empty(field, field_placeholder,field_name=None):
    empty_fields = []

    # if not all_field_filled:  # IF there are blank fields, iterate and produce error message
    for key, value in field.items():
        if value == "":
            field_placeholder[key].error(
                f"Please do not leave field blank")
            empty_fields.append(key)

        else:
            pass

    return not empty_fields


def activation_page(user=user, conn=conn, layout='centered'):  # activation page for new users

    psd_place = {}
    psd = {}

    st.write("## __User Account Activation__")
    st.markdown("___")
    activation_place = st.empty()
    if layout == 'wide':
        col1, col2, col3 = activation_place.beta_columns([1, 3, 1])
    else:
        col2 = activation_place

    with col2.form(key="activate", clear_on_submit=True):
        # psd 1
        psd["first"] = st.text_input(label="New Password", key="new_password",
                                     type="password", help="Enter new password")
        psd_place["first"] = st.empty()

        # psd re-enter
        psd["second"] = st.text_input(label="Re-enter Password", key="re_enter_password",
                                      type="password", help="Re-enter new password")
        psd_place["second"] = st.empty()

        submit_activation = st.form_submit_button(
            "Confirm", help="Submit desired password to activate user account")
        if submit_activation:

            has_submitted = check_if_field_empty(psd, psd_place)
            if has_submitted:
                if psd["first"] == psd["second"]:
                    user["psd"] = psd["first"]
                    user["status"] = "ACTIVE"
                    
                    #   TODO
                    #  update_psd(user, conn)
                    st.success(""" Successfully activated account. 
                                Please return to Login Page. 
                                """)
                    user = {}
                else:
                    st.error(
                        "Activation failed, passwords does not match. Please enter again.")
    has_submit_back = col2.button(label="Back", key="back_to_login")
    if has_submit_back:
        login_page(layout)


def login_page(layout='centered'):

    login_place = st.empty()  # PLACEHOLDER to replace with error message

    # >>>> Place login container at the centre when layout == 'wide'
    if layout == 'wide':
        left, mid_login, right = login_place.beta_columns([1, 3, 1])
    else: # Place login container at the centre when layout =='centred'
        mid_login = login_place

    with mid_login.form(key="login", clear_on_submit=True):

       
        st.write("## Login")

        # USERNAME
        user["username"] = st.text_input(
            "Username",  key="username", help="Enter your username")
        login_field_place["username"]=st.empty()
        # PASSWORD
        user["psd"] = st.text_input("Password", value="",
                              key="safe", type="password")
        login_field_place["psd"]=st.empty()

        submit_login = st.form_submit_button("Log In")
        # st.write(f"{username},{pswrd}")

        if "attempt" not in SessionState:
            SessionState.attempt=0

        success_place = st.empty()
        if submit_login: #Verify user credentials with database
            has_submitted = check_if_field_empty(user, login_field_place, FIELDS)

            if has_submitted:
                if user_login(user,SessionState.attempt, conn): # if User exists in database
                    # st.balloons()
                    # st.header("Welcome In")
                    success_place.success("Welcome in")
                    success_side = st.sidebar.empty()

                    success_side.write("# Welcome In 👋")
                    sleep(1)
                    success_place.empty()
                    # success_side.empty()
                    with main_place:
                        dashboard.write()

                elif pswrd:
                    st.error(
                        "User entered wrong username or password. Please enter again.")


def show():
     # >>>> START >>>>
     # >>>> START >>>>
    with st.sidebar.beta_container():

        st.image("resources/MSF-logo.gif", use_column_width=True)
    # with st.beta_container():
        st.title("Integrated Vision Inspection System", anchor='title')

        st.header(
            "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
        st.markdown("""___""")
    # with st.beta_container():
    with st.beta_container():
        st.title("")
        st.title("")
        st.title("")
        
    
    # st.markdown("""___""")

    # <<<< START <<<<
    login_page(layout)
    chdir_root()  # change to root directory
    # activation_page(user,conn,layout)


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        show()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
