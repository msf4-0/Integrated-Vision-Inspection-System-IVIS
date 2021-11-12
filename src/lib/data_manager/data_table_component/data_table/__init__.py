
""" 
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
from typing import Callable, Optional, Tuple, List, Dict
import streamlit as st
layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                        page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)
import streamlit.components.v1 as components
from streamlit import session_state as session_state

SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

# >>>> User-defined Modules >>>>
from core.utils.helper import check_args_kwargs
from core.utils.log import logger

_RELEASE = True


if not _RELEASE:

    _component_func = components.declare_component(
        "data_table", url="http://localhost:3000",)
else:

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component(
        "data_table", path=build_dir)


def data_table(
    rows: List[Dict],
    columns: List[Dict],
    checkbox: bool = True,
    key: str = None,
    on_change: Optional[Callable] = None,
    args: Optional[Tuple] = None,
    kwargs: Optional[Dict] = None
) -> List[int]:
    """Generate Data Table using Material UI Data Grid MIT

    Args:
        rows (List[Dict]): Row data
        columns (List[Dict]): Table column configuration
        checkbox (bool, optional): Flag to enable checkboxes to the first column. Defaults to True.
        key (str, optional): Unique key for widget. Defaults to None.
        on_change (Optional[Callable], optional): Callback function when there are changes to the selection. Defaults to None.
        args (Optional[Tuple], optional): Callback function *args. Defaults to None.
        kwargs (Optional[Dict], optional): Callback function **kwargs. Defaults to None.

    Returns:
        List[int]: List of selected IDs
    """

    component_value = _component_func(
        rows=rows, columns=columns, checkbox=checkbox, key=key, default=[])
    logger.debug("Inside data table function")
    # create unique key for previous value
    _prev_value_name = str(key) + "_prev_value"
    if _prev_value_name not in session_state:
        session_state[_prev_value_name] = None
    logger.debug(
        f"Current: {session_state[key]}; Prev: {session_state[_prev_value_name]}")

    # if session_state[_prev_value_name] != session_state[key]:
    #     """Should run method"""
    logger.debug(session_state[_prev_value_name] != component_value)

    if component_value and (session_state[_prev_value_name] != component_value):
        if on_change:
            logger.debug("Inside callback")
            wildcard = args if args else kwargs
            if args or kwargs:
                check_args_kwargs(wildcards=wildcard, func=on_change)
            if args:
                on_change(*args)
            elif kwargs:
                on_change(*kwargs)
            else:
                on_change()

    # else:
    #     """nothing"""

    try:
        session_state[_prev_value_name] = component_value
        # st.experimental_rerun()
    except:
        pass

    return component_value


# **********************************TEMP**********************************************
# if not _RELEASE:
#     import streamlit as st
#     from streamlit import session_state as session_state

#     import pandas as pd
#     from pathlib import Path
#     SRC = Path(__file__).resolve().parents[5]  # ROOT folder -> ./src
#     LIB_PATH = SRC / "lib"

#     if str(LIB_PATH) not in sys.path:
#         sys.path.insert(0, str(LIB_PATH))  # ./lib
#     else:
#         pass

#     from core.utils.log import logger.info, log_error  # logger
#     from data_manager.database_manager import init_connection, db_fetchone
#     from data_manager.dataset_management import Dataset

#     st.subheader("Dev Data Table with Material-UI")

# # Row format
#     rows = [
#         {'id': 'sfsd', 'lastName': "Snow", 'firstName': "Jon", 'age': 35},
#         {'id': 2, 'lastName': "Lannister", 'firstName': "Cersei", 'age': 42},
#         {'id': 3, 'lastName': "Lannister", 'firstName': "Jaime", 'age': 45},
#         {'id': 4, 'lastName': "Stark", 'firstName': "Arya", 'age': 16},
#         {'id': 5, 'lastName': "Targaryen", 'firstName': "Daenerys", 'age': None},
#         {'id': 6, 'lastName': "Melisandre", 'firstName': None, 'age': 150},
#         {'id': 7, 'lastName': "Clifford", 'firstName': "Ferrara", 'age': 44},
#         {'id': 8, 'lastName': "Frances", 'firstName': "Rossini", 'age': 36},
#         {'id': 9, 'lastName': "Roxie", 'firstName': "Harvey", 'age': 65},
#         {'id': 10, 'lastName': "Roxie", 'firstName': "Harvey",
#             'age': 652222222222222222222222222222222222222222222222222222222222222222222222222222222222222},
#         {'id': 11, 'lastName': "Roxie", 'firstName': "Harvey", 'age': 65},
#         {'id': 12, 'lastName': "Roxie", 'firstName': "Harvey", 'age': 65},
#         {'id': 13, 'lastName': "Roxie", 'firstName': "Harvey", 'age': 65},
#         {'id': 14, 'lastName': "Roxie", 'firstName': "Harvey", 'age': 65},
#         {'id': 15, 'lastName': "Roxie", 'firstName': "Harvey", 'age': 65},
#         {'id': 16, 'lastName': "Roxie", 'firstName': "Harvey", 'age': 65},
#         {'id': 17, 'lastName': "Roxie", 'firstName': "Harvey", 'age': 65},
#         {'id': 18, 'lastName': "Chae", 'firstName': "Rose", 'age': 26}
#     ]

#     # Column format
#     columns = [
#         {
#             'field': "id",
#             'headerName': "ID",
#             'headerAlign': "center",
#             'align': "center",
#             'flex': 20,
#             'hideSortIcons': True,

#         },
#         {
#             'field': "firstName",
#             'headerName': "First name",
#             'headerAlign': "center",
#             'align': "center",
#             'flex': 150,
#             'hideSortIcons': True,
#         },
#         {
#             'field': "lastName",
#             'headerName': "Last name",
#             'headerAlign': "center",
#             'align': "center",
#             'flex': 150,
#             'hideSortIcons': True,
#         },
#         {
#             'field': "age",
#             'headerName': "Age",
#             'headerAlign': "center",
#             'align': "center",
#             'type': "number",
#             'hideSortIcons': True,

#             'flex': 50,
#             'resizable': True,
#         },
#         {
#             'field': "fullName",
#             'headerName': "Full name",
#             'description': "This column has a value getter and is not sortable.",
#             'headerAlign': "center",
#             'align': "left",
#             'hideSortIcons': True,
#             'sortable': False,

#             'flex': 100,
#             'resizable': True,

#         },
#     ]

#     rows = data_table(rows, columns, key='test_table', checkbox=False)
#     if rows:
#         st.write("You have selected", rows)
