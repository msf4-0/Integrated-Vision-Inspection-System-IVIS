"""
Title: 
Date: 
Author: 
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
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state

# DEFINE Web APP page configuration
layout = "wide"
st.set_page_config(
    page_title="Integrated Vision Inspection System",
    page_icon="static/media/shrdc_image/shrdc_logo.png",
    layout=layout,
)

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib


# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
from data_manager.database_manager import init_connection

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration <<<<
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])
# <<<< Variable Declaration <<<<
# >>>> Template >>>>
chdir_root()  # change to root directory

with st.sidebar.container():
    st.image("resources/MSF-logo.gif", use_column_width=True)
    # with st.beta_container():
    st.title("Integrated Vision Inspection System", anchor="title")
    st.header(
        "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor="heading"
    )
    st.markdown("""___""")
    # <<<< Template <<<<


def main():
    # two list of elements
    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5, 6, 7]

    # place both list in a List
    if "test_list" not in session_state:
        session_state.test_list = [x, y]

    # define session_state variable to store the switched list
    if "switch_list" not in session_state:
        session_state.switch_list = session_state.test_list[0]

    # callback function to switch the list
    def switch_list():
        session_state.switch_list = session_state.test_list.pop(0)
        session_state.test_list.append(session_state.switch_list)
        log_info(session_state.switch_list)

    st.multiselect(label="switch list test",
                   options=session_state.switch_list)

    st.button(label="switch list toggle",
              on_click=switch_list)


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
