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
from time import sleep
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state

# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from core.utils.log import logger
from user.user_management import AccountStatus, User, query_all_users
from main_page_management import MainPagination, UserManagementPagination, reset_user_management_page
from data_manager.data_table_component.data_table import data_table
from pages.sub_pages.user_page import create_new_user


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

    # TODO: allow reset user passwords
    # TODO: allow user deletion or role change

    def to_new_user_cb():
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

    selected_user_row = next(
        filter(lambda x: x['id'] == selected_user_ids[0], users))
    info_col1, _, info_col2 = st.columns([1, 0.1, 1])
    for field1, field2 in zip(USER_FIELDS[:4], USER_FIELDS[4:]):
        with info_col1:
            if field1 != 'id':
                st.markdown(f"**{field1}**: {selected_user_row[field1]}")
        with info_col2:
            if field2 != 'id':
                st.markdown(f"**{field2}**: {selected_user_row[field2]}")

    session_state.current_user = User(selected_user_ids[0])

    def to_edit_user_cb():
        # session_state.current_user is required to modify existing user info
        session_state.user_manage_pagination = UserManagementPagination.EditUser

    st.button("Edit selected user's info", key='btn_edit_selected_user',
              on_click=to_edit_user_cb)

    st.button("Reset selected user's password", key='btn_reset_psd')

    all_status = AccountStatus.get_all_status()
    current_user_status_idx = all_status.index(
        session_state.current_user.account_status.name)
    st.markdown("#### Edit selected user's status")
    selected_user_status = st.selectbox(
        "Select a user status", options=all_status,
        index=current_user_status_idx, key='selected_user_status')
    if st.button("Confirm change", key='btn_confirm_change_status'):
        new_status = AccountStatus.from_string(selected_user_status)
        session_state.current_user.update_status(new_status)
        st.success("User status updated successfully")
        sleep(1)
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
