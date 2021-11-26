"""
Title: Page for checking/editing info of the currently logged in user
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
from itertools import cycle
from pathlib import Path
import pandas as pd
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state
from main_page_management import MainPagination

# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from core.utils.log import logger
from user.user_management import query_user_session_log


def main():
    logger.debug("Navigator: User Info")
    st.header("User Information")
    st.markdown("___")

    attr2fullname = {
        'emp_id': 'Employee Company ID',
        'first_name': 'First Name',
        'last_name': 'Last Name',
        'email': 'Employee Company Email',
        'department': 'Department',
        'position': 'Position',
        'username': 'Username',
        'role': 'Role'}
    user_attrs = list(attr2fullname)

    user_info = session_state.user.__dict__

    info_col1, _, info_col2 = st.columns([1, 0.1, 1])
    cols = cycle((info_col1, info_col2))
    for (attr, fullname), col in zip(attr2fullname.items(), cols):
        with col:
            st.markdown(f"**{fullname}**: {user_info[attr]}")

    def to_edit_user_cb():
        # session_state.current_user is required to modify existing user info
        session_state.current_user = session_state.user
        session_state.main_pagination = MainPagination.CreateUser

    st.button("Edit user info", key='btn_edit_logged_in_user',
              on_click=to_edit_user_cb)

    st.markdown("___")
    st.header("User Session Log")
    st.markdown("___")
    user_session_log, column_names = query_user_session_log(
        session_state.user.id, return_dict=True)
    session_df = pd.DataFrame(user_session_log, columns=column_names)
    session_df.fillna('-', inplace=True)
    st.dataframe(session_df, width=1000, height=1200)


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
