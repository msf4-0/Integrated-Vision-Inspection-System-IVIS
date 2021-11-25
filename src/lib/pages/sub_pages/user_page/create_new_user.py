"""
Title: Create New User Page
Date: 27/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
import sys
from pathlib import Path

import streamlit as st
from streamlit import session_state
from streamlit import cli as stcli
# -------------

SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
# TEST_MODULE_PATH = SRC / "test" / "test_page" / "module"
if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

from data_manager.database_manager import init_connection
from user.user_management import USER_ROLES, User, UserRole, check_if_other_user_exists, check_if_user_exists, create_user, check_if_field_empty, get_default_user_info, verify_password
from core.utils.code_generator import make_random_password
from core.utils.log import logger

# >>>> Variable declaration >>>>
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
# conn = psycopg2.connect(
#     "host=localhost port=5432 dbname=eye user=shrdc password=shrdc")
# st.write(st.secrets["postgres"])
conn = init_connection(**st.secrets["postgres"])
# layout = 'centered'
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<


def show(layout='wide'):

    new_user = {}
    place = {}
    has_submitted = False

    # st.write(session_state.get('current_user'))
    # st.write(f"{type(session_state.get('current_user')) = }")
    # st.write(f"{isinstance(session_state.get('current_user'), User) = }")

    if 'current_user' in session_state and \
            isinstance(session_state.current_user, User):
        # using this session_state to denote whether it is an existing user
        existing_user = session_state.current_user
        logger.debug("Navigator: Edit User Info__")
        st.markdown("## __Edit User Info__")
    else:
        existing_user = None
        st.markdown("## __Add New User__")
    st.markdown("___")

    if existing_user:
        # to use existing values from the user
        current_user = existing_user
    else:
        # use fields with default values
        current_user = get_default_user_info()

    if layout == 'wide':
        _, col2, _ = st.columns([1, 3, 1])
    else:
        col2 = st.container()
    # placeholder=col2.empty()
    # with st.container():
    with col2.form(key="create", clear_on_submit=False):
        st.write("### Employee Company Details")

        new_user["emp_id"] = st.text_input(
            "Employee Company ID", key="emp_id",
            value=current_user.emp_id, help="Enter Employee Company ID")
        place["emp_id"] = st.empty()

        new_user["first_name"] = st.text_input(
            "First Name", key="first_name",
            value=current_user.first_name, help="Enter First Name of Employee")
        place["first_name"] = st.empty()

        new_user["last_name"] = st.text_input(
            "Last Name", key="last_name",
            value=current_user.last_name, help="Enter Last Name of Employee")
        place["last_name"] = st.empty()

        new_user["email"] = st.text_input(
            "Employee Company Email", key="email",
            value=current_user.email, help="Enter Employee Company Email")
        place["email"] = st.empty()

        new_user["department"] = st.text_input(
            "Department", key="department",
            value=current_user.department, help="Enter Company Deparment Employee's in")
        place["department"] = st.empty()

        new_user["position"] = st.text_input(
            "Position", key="position",
            value=current_user.position, help="Enter Employee's Job Position")
        place["position"] = st.empty()

        st.markdown("___")
        st.write("### Account Details")

        new_user["username"] = st.text_input(
            "Username", key="username",
            value=current_user.username, help="Enter your username")
        place["username"] = st.empty()

        if existing_user:
            role_name = current_user.role.fullname
            if session_state.user.role != UserRole.Administrator:
                # skip admin if current session's user is not Admin
                role_options = USER_ROLES[1:]
            else:
                role_options = USER_ROLES
        else:
            # new user defaults to Admin role on first render as only the Admin
            # can create new user
            role_name = UserRole.Administrator.fullname
            role_options = USER_ROLES
        role_idx = role_options.index(role_name)
        new_user["role"] = st.selectbox(
            "Role", options=role_options, key="role",
            index=role_idx, help="""Enter the role of the Employee in this project""")
        with st.expander("Details of Role", expanded=False):
            role_details = """
            ### __Roles__: \n
            1. {0}: Full access to tool and manage user registration and authentication. \n
            2. {1}: Full access to tool (Annotation, Model Training, Deployment) \n
            3. {2}: Limited access to tool (Annotation, Model Training) \n
            4. {3}: Only has access to Annotation tools
            """ .format(USER_ROLES[0], USER_ROLES[1], USER_ROLES[2], USER_ROLES[3])

            st.info(role_details)
            st.warning(
                """NOTE: If you are creating for a non-administrator user, you will
                need to provide the temporary password here to the user to activate 
                the account first before the user can use the account to login.""")
        place["role"] = st.empty()

        if existing_user:
            new_user["existing_psd"] = st.text_input(
                "Current Password", key="existing_psd", type="password")
            place['existing_psd'] = st.empty()
            psd_label = "New Password"
        else:
            psd_label = "Password"

        new_user["psd"] = st.text_input(
            psd_label, key="safe", type="password")
        place["psd"] = st.empty()
        new_user["confirm_psd"] = st.text_input(
            "Confirm Password", key="confirm_psd", type="password")
        place['confirm_psd'] = st.empty()

        if not existing_user:
            psd_gen_flag = st.checkbox(
                "Or Use Auto-generated Password (Set as default when checked)",
                key="auto_gen_psd_checkbox")
            random_psd = make_random_password(22)
            st.write(
                """
                    | {0} |
                    |--------|

                    """.format(random_psd))
            if psd_gen_flag:
                new_user["psd"] = random_psd
                new_user["confirm_psd"] = random_psd

        st.markdown("___")
        submit_create_user_form = st.form_submit_button(
            "Submit")
        # submit_create_user_form = st.button(
        #     "Submit", key="submit_create_user_form")
        if submit_create_user_form:  # IF all fields are populated -> RETURN True
            if existing_user:
                all_fields = FIELDS.copy()
                all_fields['existing_psd'] = "Current Password"
            else:
                all_fields = FIELDS
            has_submitted = check_if_field_empty(new_user, place, all_fields)
            if not has_submitted:
                st.stop()

            if not existing_user:
                # only two unique fields/columns in the 'users' table
                unique_field2value = {
                    'emp_id': new_user["emp_id"],
                    'username': new_user["username"]
                }
                exists_flag, columns_with_used_values = check_if_user_exists(
                    unique_field2value, conn)
            else:
                psd_correct = verify_password(
                    current_user, new_user['existing_psd'])
                if not psd_correct:
                    place['existing_psd'].error("Incorrect password!")
                    logger.error(f"Incorrect password for username: "
                                 f"{current_user.username}")
                    st.stop()

                exists_flag, columns_with_used_values = check_if_other_user_exists(
                    new_user["emp_id"], new_user["username"], current_user, conn)

            if exists_flag:
                for col in columns_with_used_values:
                    fieldname = all_fields[col]
                    place[col].error(
                        f"This **{fieldname}** already exists, a unique value is required.")
                st.stop()

            if new_user["confirm_psd"] != new_user['psd']:
                place['confirm_psd'].error(
                    "Password is not the same as above!")
                st.stop()

            if existing_user:
                # note that this `current_user` is referenced to session_state.current_user
                current_user.update_info(new_user)
                st.success("Information updated successfully")
                logger.info(f"Information updated successfully for user with "
                            f"username: {current_user.username}")
                # stop here as the rest of the code is for creating new user
                st.stop()

            create_success = create_user(new_user, conn)
            if not create_success:
                st.error("Error creating user!")
                st.stop()
            st.success(f"Successfully created new user: '{new_user['username']}' "
                       f"with the role of '{new_user['role']}'")
            logger.info(f"Successfully created new user: '{new_user['username']}' "
                        f"with the role of '{new_user['role']}'")

            if new_user['role'] != "Administrator":
                st.warning(
                    f"""Please advice the user to login with the temporary password used
                    here to activate the account.\n__Employee Temporary Password: 
                    {new_user["psd"]}__
                    """)
            else:
                # need to clear cache to allow query_all_admins() to rerun
                st.legacy_caching.clear_cache()

    return new_user, has_submitted


def main():
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

        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
