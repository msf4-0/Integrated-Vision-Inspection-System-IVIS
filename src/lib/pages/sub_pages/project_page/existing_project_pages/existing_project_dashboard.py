"""
Title: Existing Project Dashboard
Date: 19/8/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
from enum import IntEnum
from time import sleep
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state as session_state

# DEFINE Web APP page configuration
# layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[5]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
from core.utils.helper import create_dataframe, get_df_row_highlight_color, get_textColor, current_page, non_current_page
from core.utils.form_manager import remove_newline_trailing_whitespace
from data_manager.database_manager import init_connection
from data_manager.dataset_management import NewDataset, query_dataset_list, get_dataset_name_list
from project.project_management import ExistingProjectPagination, ProjectPermission, Project
from data_editor.editor_management import Editor
from data_editor.editor_config import editor_config
from pages.sub_pages.dataset_page.new_dataset import new_dataset

# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


# >>>> Variable Declaration >>>>
new_project = {}  # store
place = {}

chdir_root()  # change to root directory


def dashboard():
    st.write(f"## **Overview:**")
    # TODO #79 Add dashboard to show types of labels and number of datasets
    # >>>>>>>>>>PANDAS DATAFRAME for LABEL DETAILS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    df = session_state.project.editor.create_table_of_labels()
    df.index.name = 'No.'
    df['Percentile (%)'] = df['Percentile (%)'].map("{:.2f}".format)
    styler = df.style

    # >>>> Annotation table placeholders
    annotation_col1, annotation_col2 = st.columns([3, 0.5])

    annotation_col1.write("### **Annotations**")
    annotation_col2.write(
        f"### Total labels: {len(session_state.project.editor.labels_results)}")

    st.table(styler.set_properties(**{'text-align': 'center'}).set_table_styles(
        [dict(selector='th', props=[('text-align', 'center')])]))

    # >>>>>>>>>>PANDAS DATAFRAME for LABEL DETAILS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # >>>>>>>>>>PANDAS DATAFRAME for DATASET DETAILS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    df = create_dataframe(session_state.project.datasets, column_names=session_state.project.column_names,
                          sort=True, sort_by='ID', asc=True, date_time_format=True)
    df_loc = df.loc[:, "ID":"Date/Time"]

    styler = df_loc.style

    # >>>> Dataset table placeholders
    dataset_table_col1, dataset_table_col2 = st.columns([3, 0.5])

    dataset_table_col1.write("### **Datasets**")
    dataset_table_col2.write(
        f"### Total datasets: {len(session_state.project.datasets)}")

    st.table(styler.set_properties(**{'text-align': 'center'}).set_table_styles(
        [dict(selector='th', props=[('text-align', 'center')])]))    # >>>>>>>>>>PANDAS DATAFRAME for DATASET DETAILS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    st.write(vars(session_state.project))

if __name__ == "__main__":
    if st._is_running_with_streamlit:
        dashboard()

    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
