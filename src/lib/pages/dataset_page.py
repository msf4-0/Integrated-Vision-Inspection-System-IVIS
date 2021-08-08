"""
Title:Dataset Dashboard Page
Date: 1/8/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
import streamlit as st
from collections import namedtuple
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state

# DEFINE Web APP page configuration
layout = 'wide'
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
TEST_MODULE_PATH = SRC / "test" / "data_table_component" / "data_table"
DATA_DIR = Path.home() / '.local/share/integrated-vision-inspection-system/app_media'

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

if str(TEST_MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(TEST_MODULE_PATH))  # ./lib
else:
    pass
# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
from core.utils.helper import create_dataframe
from data_manager.database_manager import init_connection
from data_manager.dataset_management import Dataset, get_dataset_name_list, query_dataset_list
from data_table import data_table
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration <<<<
conn = init_connection(**st.secrets["postgres"])
PAGE_OPTIONS = {"Dataset", "Project", "Deployment"}
# <<<< Variable Declaration <<<<
# >>>> Template >>>>
chdir_root()  # change to root directory
# initialise connection to Database

with st.sidebar.beta_container():
    st.image("resources/MSF-logo.gif", use_column_width=True)
# with st.beta_container():
    st.title("Integrated Vision Inspection System", anchor='title')
    st.header(
        "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
    st.markdown("""___""")
    st.radio("", options=PAGE_OPTIONS, key="all_pages")


# <<<< Template <<<<


def show():
    # Page title
    st.write("# **Dataset**")
    st.markdown("___")

    # >>>> Dataset SIDEBAR >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    DATASET_PAGE = ("All Datasets", "New Dataset")
    with st.sidebar.beta_expander("Dataset Page", expanded=True):
        session_state.current_page = st.radio("", options=DATASET_PAGE,
                                              index=0)

    # <<<< Dataset SIDEBAR <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    if "dataset_data" not in session_state:
        # session_state.existing_dataset = []
        session_state.dataset_data = {}
    if "data_name_list" not in session_state:
        session_state.data_name_list = {}

    # ********* QUERY DATASET ********************************************
    existing_dataset, dataset_table_column_names = query_dataset_list()
    dataset_dict = get_dataset_name_list(existing_dataset)
    # ********* QUERY DATASET ********************************************

    # ************COLUMN PLACEHOLDERS *****************************************************
    # >>>> Column for Create New button and Dataset selectbox
    topcol1, _, topcol2, _, top_col3 = st.beta_columns(
        [0.15, 0.3, 2, 0.1, 3.5])
    # topcol3 for dataset ID

    # >>>> COLUMNS for BUTTONS
    _, button_col1, _, button_col2, _, button_col3, _, desc_col4 = st.beta_columns(
        [0.115, 0.375, 0.3, 0.75, 0.3, 0.375, 0.1, 3.15])

    # >>>> Main placeholders
    outercol1, _, outercol2 = st.beta_columns(
        [2.5, 0.1, 3.5])

    # column placeholder for variable/class objects
    _, varscol1, varscol2, varscol3 = st.beta_columns([0.4, 2, 3.5, 0.15])
    # ************COLUMN PLACEHOLDERS *****************************************************

    # >>>> CREATE NEW DATASET AND SELECT DATASET >>>>>>>>>>>>>>>>>>>>>>>>>>>>
    with topcol1:
        st.write("## ")
        st.button("➕️", key="new_dataset", help="Create new dataset")

    # TODO: removed 'dataset selection'????
    with topcol2:
        dataset_selection = st.selectbox("Select Dataset", options=dataset_dict,
                                         key="dataset_sel", help="Select dataset to view details")

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>TABLE OF DATA>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # >>>>>>>>> INSTATIATE DATASET CLASS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # def instantiate_dataset_class(dataset_dict):
    if "dataset" not in session_state:
        session_state.dataset = Dataset(
            dataset_dict[session_state.dataset_sel])
    else:
        session_state.dataset = Dataset(
            dataset_dict[session_state.dataset_sel])
    # varscol1.write(vars(session_state.dataset))

    session_state.data_name_list = session_state.dataset.get_data_name_list(
        session_state.data_name_list)
    # st.write(session_state.data_name_list)
    # <<<<<<<< INSTATIATE DATASET CLASS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>> BUTTON >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # ****** PAGINATION CALLBACK ********
    if "dataset_table_page" not in session_state:
        session_state.dataset_table_page = 0

    def next_dataset_table_page():
        session_state.dataset_table_page += 1
        log_info("Increment")

    def prev_dataset_table_page():
        session_state.dataset_table_page -= 1
        log_info("Decrement")

    num_data_per_page = 25
    dataset_length = len(dataset_dict)
    num_data_page = dataset_length // num_data_per_page
    # df["Task Name"]
    # >>>>>>>> BUTTON >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # >>>>>>>>>>PANDAS DATAFRAME >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    log_info("Dataframe zone")
    with outercol1:

        start = num_data_per_page * session_state.dataset_table_page
        end = start + num_data_per_page

        df = create_dataframe(existing_dataset, column_names=dataset_table_column_names,
                              sort=True, sort_by='ID', asc=True, date_time_format=True)
        # Loc columns from ID to Date/Time
        df_loc = df.loc[:, "ID":"Date/Time"]
        df_slice = df_loc.iloc[start:end]
        # if data_selection:

        def highlight_row(x, selections):

            if x.Name in selections:

                return ['background-color: #90a4ae'] * len(x)
            else:
                return ['background-color: '] * len(x)

        styler = df_slice.style.apply(
            highlight_row, selections=session_state.dataset_sel, axis=1)

        # else:
        #     styler = df_slice.style

        st.dataframe(styler.set_properties(**{'text-align': 'center'}).set_table_styles(
            [dict(selector='th', props=[('text-align', 'center')])]))

    # <<<<<<<<<<PANDAS DATAFRAME <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    if num_data_page > 1:
        if session_state.dataset_table_page < num_data_page:
            button_col3.button(">", on_click=next_dataset_table_page)
        else:
            # this makes the empty column show up on mobile
            button_col3.write("")

        if session_state.dataset_table_page > 0:
            button_col1.button("<", on_click=prev_dataset_table_page)
        else:
            # this makes the empty column show up on mobile
            button_col1.write("")

        # button_col2.write(
        #     f"Page {1+session_state.dataset_table_page} of {num_data_page}")
    button_col2.write(
        f"### Data **{1+start}-{len(df_slice)}** of **{dataset_length}**")  # Example <Data 1-3 of 3>
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<TABLE OF DATA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>> LOAD EXISTING DATASET >>>>>>>>>>>>>>>>>>>>>>>>#
    # TODO #17 Load details of existing dataset

    log_info("Outercol2 Zone")
    with top_col3:
        # DATASET ID
        st.write(
            f"### __Dataset ID:__ {session_state.dataset.dataset_id}")
        # TITLE
        st.write(f"# {session_state.dataset.name}")

    with outercol2:

        # DESCRIPTION
        desc_col4.write("### **Description:**")
        if session_state.dataset.desc:
            st.write(f"{session_state.dataset.desc}")
        else:
            st.write(f"No description")

        # DATA TABLE

    # **************** DATA TABLE COLUMN CONFIG ****************************
        dataset_columns = [
            {
                'field': "id",
                'headerName': "ID / Name",
                'headerAlign': "center",
                'align': "center",
                'flex': 150,
                'hideSortIcons': True,

            },
            {
                'field': "filetype",
                'headerName': "File Type",
                'headerAlign': "center",
                'align': "center",
                'flex': 150,
                'hideSortIcons': True,
            },
            {
                'field': "created",
                'headerName': "Created At",
                'headerAlign': "center",
                'align': "center",
                'flex': 150,
                'hideSortIcons': True,
                'type': 'date',
            },

        ]
    # **************** DATA TABLE COLUMN CONFIG ****************************

        selection = data_table(
            session_state.dataset.data_name_list, dataset_columns, key="data_table")
        st.write(f"Selection")
        st.write(selection)

    # TODO #44 Image viewer and data control features to List view and Gallery
    # Add deletion
    # Add view image
    #Add Edit callback
    

    # TODO #18

    # <<<<<<<<<<<<<<<<<<<<<<<<<< LOAD EXISTING DATASET <<<<<<<<<<<<<<<<<<<<<#
    # TODO: ADD show() function to load existing dataset and new dataset page?
    # st.write(dataset_dict)
    pass


def main():
    show()


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
