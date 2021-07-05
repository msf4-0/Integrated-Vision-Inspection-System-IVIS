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
import logging
import psycopg2
# -------------

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
sys.path.insert(0, str(Path(SRC, 'lib')))  # ./lib
# print(sys.path[0])
sys.path.insert(0, str(Path(Path(__file__).parent, 'module')))

#--------------------Logger-------------------------#

FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
DATEFMT = '%d-%b-%y %H:%M:%S'

# logging.basicConfig(filename='test.log',filemode='w',format=FORMAT, level=logging.DEBUG)
logging.basicConfig(format=FORMAT, level=logging.INFO,
                    stream=sys.stdout, datefmt=DATEFMT)

log = logging.getLogger()

#----------------------------------------------------#
from user_management import create_user, check_if_field_empty
from code_generator import make_random_password
# Exception handling:
# if __package__ is None (module run by filename), run "import dashboard"
# if __package__ is __name__ (name of package == "pages"), run "from . import dashboard"
# try:
#     from pages import dashboard
# except:
#     PACKAGE = Path(__file__).parent  # Package folder
#     sys.path.insert(0, str(PACKAGE))

# PARENT = Path.home()  # Parent folder

ROLES = ["Annotator", "Developer (Model Training)",
         "Developer (Deployment)", "Administrator"]
FIELDS = {
    'emp_id': 'Employee Company ID',
    'first_name': 'First Name',
    'last_name': 'Last Name',
    'email': 'Employee Company Email',
    'department': 'Department',
    'position': 'Position',
    'username': 'Username',
    'role': 'Role',
    'psd': 'Password'}

# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
conn = psycopg2.connect(
    "host=localhost port=5432 dbname=eye user=shrdc password=shrdc")
layout = 'centered'
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<


def show(layout='centered'):

    new_user = {}
    place = {}
    has_submitted = False

    # with st.form("create_new_user", clear_on_submit=True):
    st.write("## __Add New User__")
    st.markdown("___")
    create_user_place = st.empty()
    if layout == 'wide':
        col1, col2, col3 = create_user_place.beta_columns([1, 3, 1])
    else:
        col2 = create_user_place
    # placeholder=col2.empty()
    # with st.beta_container():
    random_psd = make_random_password(22)
    with col2.form(key="create", clear_on_submit=True):
        st.write("### Employee Company Details")

        new_user["emp_id"] = st.text_input(
            "Employee Company ID", key="emp_id", help="Enter Employee Company ID")
        place["emp_id"] = st.empty()

        new_user["first_name"] = st.text_input(
            "First Name", key="first_name", help="Enter First Name of Employee")
        place["first_name"] = st.empty()

        new_user["last_name"] = st.text_input(
            "Last Name", key="last_name", help="Enter Last Name of Employee")
        place["last_name"] = st.empty()

        new_user["email"] = st.text_input(
            "Employee Company Email", key="email", help="Enter Employee Company Email")
        place["email"] = st.empty()

        new_user["department"] = st.text_input(
            "Department", key="department", help="Enter Company Deparment Employee's in")
        place["department"] = st.empty()

        new_user["position"] = st.text_input(
            "Position", key="position", help="Enter Employee's Job Position")
        place["position"] = st.empty()

        st.markdown("___")
        st.write("### Account Details")

        new_user["username"] = st.text_input(
            "Username", key="username", help="Enter your username")
        place["username"] = st.empty()

        new_user["role"] = st.selectbox(
            "Role", options=ROLES, key="role", help="""Enter the role of the Employee in this project""")
        with st.beta_expander("Details of Role", expanded=False):
            role_details = """
            ### __Roles__: \n
            1. {0}: Full access to tool and manage user registration and authentication. \n
            2. {1}: Full access to tool (Annotation, Model Training, Deployment) \n
            3. {2}: Limited access to tool (Annotation, Model Training) \n
            4. {3}: Only has access to Annotation tools
            """ .format(ROLES[0], ROLES[1], ROLES[2], ROLES[3])

            st.info(role_details)
        place["role"] = st.empty()

        psd_place = st.empty()
        new_user["psd"] = psd_place.text_input("Password",
                                               key="safe", type="password")
        place["psd"] = st.empty()
        psd_gen_flag = st.checkbox(
            "Auto-generated Password (Set as default when checked)", key="auto_gen_psd_checkbox")
        st.write(
            """
                | {0} |
                |--------|

                """.format(random_psd))
        if psd_gen_flag:
            # ----------Need add password generator
            new_user["psd"] = random_psd

            # st.write(
            #     """
            #     | {psd} |
            #     |--------|

            #     """.format(**new_user))
            # -------

            # psd_place.write(new_user["psd"])
            # psd_place.text_input("Password", value=new_user["psd"],
            #                      key="auto_psd_gen", type="password")

            # submit_login = st.form_submit_button("Submit")

        st.markdown("___")
        submit_create_user_form = st.form_submit_button(
            "Submit")
        # submit_create_user_form = st.button(
        #     "Submit", key="submit_create_user_form")
        if submit_create_user_form:  # IF all fields are populated -> RETURN True

            has_submitted = check_if_field_empty(new_user, place, FIELDS)

            if has_submitted:

                create_user(new_user, conn)
                # create_success_page(new_user, placeholder=create_user_place)
                # with create_user_place:
                #     st.success(""" ## __Successfully__ created new user: {0}. \n
                #             ### Please advice user to activate account with the temporary password sent to employee's company email.
                #             """.format(new_user["username"]))
                # create_user_place.write("Hi")
                st.success(""" Successfully created new user: {0}. 
                            Please advice user to activate account with the temporary password sent to employee's company email. 
                            __Employee Temporary Password: {1}__
                            """.format(new_user["username"], new_user["psd"]))
    st.write(new_user)
    return new_user, has_submitted


def write():
    # PAGES = {
    #     "CREATE_USER": create_user_page,
    #     "SUCCESS": create_success_page

    # }
    # new_user, has_submitted = PAGES["CREATE_USER"]()
    # if has_submitted:
    #     create_success_page(new_user,)
    # st.write("Hi")
    show()


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        write()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
