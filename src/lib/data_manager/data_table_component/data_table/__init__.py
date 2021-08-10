import os
import streamlit as st
import streamlit.components.v1 as components

# >>>>>>> TEMP for Logging >>>>>>>>
import logging
import sys
FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
DATEFMT = '%d-%b-%y %H:%M:%S'

# logging.basicConfig(filename='test.log',filemode='w',format=FORMAT, level=logging.INFO)
logging.basicConfig(format=FORMAT, level=logging.INFO,
                    stream=sys.stdout, datefmt=DATEFMT)

log = logging.getLogger()


_RELEASE = True


if not _RELEASE:

    _component_func = components.declare_component(
        "data_table", url="http://localhost:3000",)
else:

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component(
        "data_table", path=build_dir)


def data_table(rows, columns, key=None):
    component_value = _component_func(
        rows=rows, columns=columns, key=key, default=[])

    return component_value


# **********************************TEMP**********************************************
# if not _RELEASE:
#     import streamlit as st
#     from streamlit import session_state as session_state
#     layout = 'wide'
#     st.set_page_config(page_title="Integrated Vision Inspection System",
#                        page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

#     import pandas as pd
#     from pathlib import Path
#     SRC = Path(__file__).resolve().parents[3]  # ROOT folder -> ./src
#     LIB_PATH = SRC / "lib"

#     if str(LIB_PATH) not in sys.path:
#         sys.path.insert(0, str(LIB_PATH))  # ./lib
#     else:
#         pass

#     from core.utils.log import log_info, log_error  # logger
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
#         {'id': 10, 'lastName': "Roxie", 'firstName': "Harvey", 'age': 65},
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

#     rows = data_table(rows, columns, key='test_table')
#     if rows:
#         st.write("You have selected", rows)
