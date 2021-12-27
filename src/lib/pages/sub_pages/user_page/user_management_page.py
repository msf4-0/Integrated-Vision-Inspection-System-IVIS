"""
Title: Page to manage all users
Date: 24/11/2021
Author: Anson Tan Chen Tung
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
import sys
from pathlib import Path
from itertools import cycle
from time import sleep

import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state

# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from core.utils.log import logger
from core.utils.code_generator import make_random_password
from project.project_management import NewProject, Project
from training.training_management import NewTraining, Training
from user.user_management import USER_ROLES, AccountStatus, User, UserRole, query_all_admins, query_all_users, reset_login_page
from main_page_management import MainPagination, UserManagementPagination, reset_user_management_page
from data_manager.data_table_component.data_table import data_table
from annotation.annotation_management import reset_editor_page
from data_manager.dataset_management import NewDataset
from deployment.deployment_management import Deployment
from pages.sub_pages.user_page import create_new_user


def danger_zone_header():
    st.markdown("""
    <h3 style='color: darkred; 
    text-decoration: underline'>
    Danger Zone
    </h3>
    """, unsafe_allow_html=True)


def dashboard():
    logger.debug("Navigator: User Management Dashboard")

    USER_FIELDS = ("Employee ID", "Full Name", "Email",
                   "Department", "Position", "Role", 'Status')
    USER_DT_COLS = [
        {
            'field': "Employee ID",
            'headerName': "ID",
            'headerAlign': "center",
            'align': "center",
            'flex': 1,
            'hideSortIcons': True,
        },
        {
            'field': "Full Name",
            'headerAlign': "center",
            'align': "center",
            'flex': 4,
            'hideSortIcons': True,
        },
        {
            'field': "Email",
            'headerAlign': "center",
            'align': "center",
            'flex': 3,
            'hideSortIcons': True,
        },
        {
            'field': "Department",
            'headerAlign': "center",
            'align': "center",
            'flex': 3,
            'hideSortIcons': True,
        },
        {
            'field': "Position",
            'headerAlign': "center",
            'align': "center",
            'flex': 3,
            'hideSortIcons': True,
        },
        {
            'field': "Role",
            'headerAlign': "center",
            'align': "center",
            'flex': 3,
            'hideSortIcons': True,
        },
        {
            'field': "Status",
            'headerAlign': "center",
            'align': "center",
            'flex': 3,
            'hideSortIcons': True,
        }
    ]

    def to_new_user_cb():
        if 'current_user' in session_state:
            # must delete this to avoid editing existing user
            del session_state['current_user']
        session_state.main_pagination = MainPagination.CreateUser

    st.button("Create New User", key='btn_create_user',
              on_click=to_new_user_cb)

    users, column_names = query_all_users(
        return_dict=True, for_data_table=True)

    selected_user_ids = data_table(
        users, USER_DT_COLS,
        checkbox=False, key='user_data_table')

    if not selected_user_ids:
        st.stop()

    try:
        selected_user_row = next(
            filter(lambda x: x['id'] == selected_user_ids[0], users))
    except StopIteration:
        logger.debug("Refreshing page after deleting user will cause data_table "
                     "to still select the previous user and throw this error. Don't mind!")
        st.stop()

    # show user info
    info_col1, _, info_col2 = st.columns([1, 0.1, 1])
    cols = cycle((info_col1, info_col2))
    for field, col in zip(USER_FIELDS, cols):
        value = selected_user_row[field]
        value = '' if value is None else value
        with col:
            st.markdown(f"**{field}**: {value}")

    # this session_state is required in case the user wants to modify existing user info
    session_state.current_user = User(selected_user_ids[0])
    selected_user: User = session_state.current_user

    # edit user info
    def to_edit_user_cb():
        # session_state.current_user is required to modify existing user info
        session_state.user_manage_pagination = UserManagementPagination.EditUser

    st.button("Edit selected user's info", key='btn_edit_selected_user',
              on_click=to_edit_user_cb)

    # reset user's password
    def reset_user_psd_cb():
        random_psd = make_random_password(22)
        selected_user.update_psd(random_psd)
        selected_user.update_status(AccountStatus.NEW)
        st.info(f"New password generated for the user: **{random_psd}**  \nPlease ask "
                "the user to login with the temporary password to activate his account.")

    if st.button("Reset selected user's password", key='btn_reset_psd'):
        reset_user_psd_cb()

    # edit user role
    current_user_role_idx = USER_ROLES.index(selected_user.role.fullname)
    st.markdown("#### Edit selected user's role")
    with st.form("form_edit_role"):
        selected_user_role = st.selectbox(
            "Select a user role", options=USER_ROLES,
            index=current_user_role_idx, key='selected_user_role')
        if st.form_submit_button("Confirm change"):
            new_role = UserRole.get_enum_from_fullname(selected_user_role)
            selected_user.update_role(new_role)
            st.success("User role updated successfully")
            sleep(0.5)
            st.experimental_rerun()

    # edit user status
    all_status = AccountStatus.get_all_status()
    current_user_status_idx = all_status.index(
        selected_user.status.name)
    st.markdown("#### Edit selected user's status")
    with st.form("form_edit_status"):
        selected_user_status = st.selectbox(
            "Select a user status", options=all_status,
            index=current_user_status_idx, key='selected_user_status')
        if st.form_submit_button("Confirm change"):
            new_status = AccountStatus.from_string(selected_user_status)
            selected_user.update_status(new_status)
            st.success("User status updated successfully")
            sleep(0.5)
            st.experimental_rerun()

    # delete user
    st.markdown("___")
    danger_zone_header()
    if st.checkbox("Delete selected user", key='cbox_delete_user',
                   help="Confirmation will be asked."):
        confirm_del = st.button("Confirm deletion?",
                                key='btn_confirm_delete_user')
        if not confirm_del:
            st.stop()

        selected_user_id = selected_user.id
        delete_success = User.delete_user(selected_user_id)
        if not delete_success:
            st.error(
                "Error deleting the selected user due to constraints with other existing "
                "data, most likely due to existing annotations/training done by the user.")
            st.stop()

        del session_state['current_user']
        st.success("User deleted successfully")
        sleep(0.5)

        # clean up the session's user if he decided to delete himself
        # NOTE: the user here can only be Admin because only Admin can access here
        if selected_user_id == session_state.user.id:
            # need to reset everything just like when logout
            del session_state['user']

            reset_login_page()
            reset_user_management_page()
            NewProject.reset_new_project_page()
            reset_editor_page()
            NewDataset.reset_new_dataset_page()
            NewTraining.reset_new_training_page()
            Training.reset_training_page()
            Project.reset_project_page()
            Project.reset_settings_page()
            Deployment.reset_deployment_page()

            if not query_all_admins():
                # no more admin available, therefore redirect to create one
                session_state.no_admin = True
                session_state.main_pagination = MainPagination.CreateUser
            else:
                session_state.main_pagination = MainPagination.Login
        st.experimental_rerun()


def main():
    st.title("User Management")
    st.markdown("___")

    user_manage_page2func = {
        UserManagementPagination.Dashboard: dashboard,
        UserManagementPagination.CreateUser: create_new_user.show,
        # EditUser will use session_state.current_user to denote it's an existing user
        UserManagementPagination.EditUser: create_new_user.show,
    }

    # ********************** SESSION STATE ******************************
    if 'user_manage_pagination' not in session_state:
        session_state.user_manage_pagination = UserManagementPagination.Dashboard

    # >>>> RETURN TO ENTRY PAGE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    if session_state.user_manage_pagination != UserManagementPagination.Dashboard:
        def back_to_dashboard_cb():
            reset_user_management_page()
            session_state.user_manage_pagination = UserManagementPagination.Dashboard

        st.sidebar.button(
            "Back to User Management Dashboard",
            key="btn_back_to_user_management_dashboard",
            on_click=back_to_dashboard_cb)

    logger.debug(
        f"Entering Page: {session_state.user_manage_pagination}")
    user_manage_page2func[session_state.user_manage_pagination]()


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
