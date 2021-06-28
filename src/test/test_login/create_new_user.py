"""
Title: Create New User Page
Date: 27/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
import streamlit as st
from streamlit import cli as stcli
import sys
from pathlib import Path
from time import sleep
import logging
import psycopg2
# -------------

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
sys.path.insert(0, str(Path(SRC, 'lib')))  # ./lib
print(sys.path[0])

#--------------------Logger-------------------------#

FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
DATEFMT = '%d-%b-%y %H:%M:%S'

# logging.basicConfig(filename='test.log',filemode='w',format=FORMAT, level=logging.DEBUG)
logging.basicConfig(format=FORMAT, level=logging.INFO,
                    stream=sys.stdout, datefmt=DATEFMT)

log = logging.getLogger()

#----------------------------------------------------#
from .user_management import create_usertable, create_user

# Exception handling:
# if __package__ is None (module run by filename), run "import dashboard"
# if __package__ is __name__ (name of package == "pages"), run "from . import dashboard"
# try:
#     from pages import dashboard
# except:
#     PACKAGE = Path(__file__).parent  # Package folder
#     sys.path.insert(0, str(PACKAGE))

# PARENT = Path.home()  # Parent folder


def write():
    new_user = {}
    with st.form("create_new_user", clear_on_submit=True):
        st.write("## Add New User")
        st.write("### Employee Company Details")
        new_user["emp_id"] = st.text_input(
            "Employee Company ID", key="emp_id", help="Enter Employee Company ID")
        new_user["first_name"] = st.text_input(
            "First Name", key="first_name", help="Enter First Name of Employee")
        new_user["last_name"] = st.text_input(
            "Last Name", key="last_name", help="Enter Last Name of Employee")
        new_user["email"] = st.text_input(
            "Employee Company Email", key="email", help="Enter Employee Company Email")
        new_user["department"] = st.text_input(
            "Department", key="department", help="Enter Company Deparment Employee's in")
        new_user["position"] = st.text_input(
            "Position", key="position", help="Enter Employee's Job Position")

        st.write("### Account Details")
        new_user["username"] = st.text_input(
            "Username", key="username", help="Enter your username")
        ROLES = ["Administrator",
                 "Developer (Deployment)", "Developer (Model Trainign)", "Annotator"]
        new_user["role"] = st.selectbox(
            "Role", key="role", help="Enter the role of the Employee in this project")
        with st.beta_expander():
            st.info("""
            #### Roles: \n
            1. Administrator: Full access to tool and manage user registration and authentication. \n
            2. Developer (Deployment): Full access to tool (Annotation, Model Training, Deployment) \n
            3. Developer (Deployment): Limited access to tool (Annotation, Model Training) \n
            4. Annotator: Only has access to Annotation tools
            """)
        psd_place = st.empty()
        new_user["psd"] = psd_place.text_input("Password",
                                               key="safe", type="password")
        psd_gen_flag = st.checkbox("Auto-generated Password")

        if psd_gen_flag:
            # ----------Need add password generator
            new_user["psd"] = SomePasswordGenerator()
            # -------
            psd_place.text_input("Password", value=new_user["psd"] 
                                 key="safe", type="password")

        submit_login = st.form_submit_button("Submit")

        return new_user


def main():
    write()


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
