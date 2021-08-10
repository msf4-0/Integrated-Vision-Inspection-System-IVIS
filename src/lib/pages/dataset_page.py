"""
Title:Dataset Dashboard Page
Date: 1/8/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
from enum import IntEnum
import streamlit as st
from collections import namedtuple
from threading import Thread
from time import sleep
from streamlit.report_thread import add_report_ctx
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
from core.utils.file_handler import delete_file_directory
from data_import.data_upload_module import data_uploader
from data_manager.database_manager import init_connection
from data_manager.dataset_management import Dataset, DataPermission, DatasetPagination, get_dataset_name_list, query_dataset_list
from pages.sub_pages.dataset_page import new_dataset
from annotation.annotation_manager import Task
from data_table import data_table
# from data_table_test import data_table # FOR DEVELOPMENT
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration <<<<
conn = init_connection(**st.secrets["postgres"])
PAGE_OPTIONS = {"Dataset", "Project", "Deployment"}
place = {}  # dictionary to store placeholders


# <<<< Variable Declaration <<<<
chdir_root()  # change to root directory

with st.sidebar.container():
    st.image("resources/MSF-logo.gif", use_column_width=True)

    st.title("Integrated Vision Inspection System", anchor='title')
    st.header(
        "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
    st.markdown("""___""")
    st.radio("", options=PAGE_OPTIONS, key="all_pages")

# <<<< Template <<<<


def dashboard():
    # Page title
    st.write("# **Dataset**")
    st.markdown("___")

    # ********* QUERY DATASET ********************************************
    existing_dataset, dataset_table_column_names = query_dataset_list()
    dataset_dict = get_dataset_name_list(existing_dataset)
    # ********* QUERY DATASET ********************************************

    if "dataset_data" not in session_state:
        # session_state.existing_dataset = []
        session_state.dataset_data = {}
    if "data_name_list" not in session_state:
        session_state.data_name_list = {}
    if 'append_data_flag' not in session_state:
        session_state.append_data_flag = DataPermission.ViewOnly
    if 'temp_name' not in session_state:
        session_state.temp_name = None
    if 'dataset_sel' not in session_state:
        session_state.dataset_sel = session_state.temp_name if session_state.temp_name else existing_dataset[
            0].Name

    # ************COLUMN PLACEHOLDERS *****************************************************
    # >>>> Column for Create New button and Dataset selectbox
    topcol1, _, topcol2, _, top_col3 = st.columns(
        [0.15, 0.3, 2, 0.1, 3.5])
    # topcol3 for dataset ID
    place["dataset_id"] = top_col3.empty()

    # >>>> COLUMNS for BUTTONS
    _, button_col1, _, button_col2, _, button_col3, _, desc_col4 = st.columns(
        [0.115, 0.375, 0.3, 0.75, 0.3, 0.375, 0.1, 3.15])

    # >>>> Main placeholders
    outercol1, _, outercol2 = st.columns(
        [2.5, 0.1, 3.5])

    place["desc"] = outercol2.empty()

    # column placeholder for variable/class objects
    _, varscol1, varscol2, varscol3 = st.columns([0.4, 2, 3.5, 0.15])
    # ************COLUMN PLACEHOLDERS *****************************************************

    # >>>> CREATE NEW DATASET AND SELECT DATASET >>>>>>>>>>>>>>>>>>>>>>>>>>>>
    with topcol1:
        def to_new_dataset_page():
            session_state.dataset_pagination = DatasetPagination.New

        st.write("## ")
        st.button("➕️", key="create_new_dataset",
                  on_click=to_new_dataset_page, help="Create new dataset")

    # TODO: removed 'dataset selection'????
    with topcol2:
        def append_data_flag_0():

            log_info("Enter append_data_flag_0 SELECTBOX")
            session_state.append_data_flag = DataPermission.ViewOnly
            session_state.selectbox_flag = 1

            if "dataset" not in session_state:
                session_state.dataset = Dataset(
                    dataset_dict[session_state.dataset_sel])
            else:
                # st.write(session_state.dataset_sel)

                session_state.dataset = Dataset(
                    dataset_dict[session_state.dataset_sel])

            if 'name' in session_state:  # RESET Text field
                del session_state.name
                del session_state.desc

        dataset_selection = st.selectbox("Select Dataset", options=dataset_dict,
                                         key="dataset_sel", on_change=append_data_flag_0, help="Select dataset to view details")

        if "dataset" not in session_state:
            session_state.dataset = Dataset(
                dataset_dict[session_state.dataset_sel])
        else:
            session_state.dataset = Dataset(
                dataset_dict[session_state.dataset_sel])    # *******************************************************************************
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>TABLE OF DATA>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # >>>>>>>>> INSTATIATE DATASET CLASS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    session_state.data_name_list = session_state.dataset.get_data_name_list(
        session_state.data_name_list)

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

        def highlight_row(x, selections):

            if x.Name in selections:

                return ['background-color: #90a4ae'] * len(x)
            else:
                return ['background-color: '] * len(x)

        styler = df_slice.style.apply(
            highlight_row, selections=session_state.dataset_sel, axis=1)

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

    button_col2.write(
        f"### Data **{1+start}-{len(df_slice)}** of **{dataset_length}**")  # Example <Data 1-3 of 3>

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<TABLE OF DATA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # *******************************************************************************

    # >>>>>>>>>>>>>>>>>>>>>>> LOAD EXISTING DATASET >>>>>>>>>>>>>>>>>>>>>>>>#

    log_info("Outercol2 Zone")

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>VIEW ONLY >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if session_state.append_data_flag == DataPermission.ViewOnly:
        with place['dataset_id'].container():
            # DATASET ID
            st.write(
                f"### __Dataset ID:__ {session_state.dataset.dataset_id}")
            # TITLE
            st.write(f"# {session_state.dataset.name}")

        with place['desc'].container():

            # DESCRIPTION
            desc_col4.write("### **Description:**")
            if session_state.dataset.desc:
                st.write(f"{session_state.dataset.desc}")
            else:
                st.write(f"No description")

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>EDIT MODE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    elif session_state.append_data_flag == DataPermission.Edit:
        def check_if_name_exist(field_placeholder, conn):
            context = ['name', session_state.name]

            # Do not check if name same as current dataset name
            if (session_state.name) and (session_state.name != session_state.dataset.name):
                if session_state.dataset.check_if_exist(context, conn):
                    field_placeholder['name'].error(
                        f"Dataset name used. Please enter a new name")

                    log_error(f"Dataset name used. Please enter a new name")
                    return False
                else:

                    # # UPDATE NAME
                    # session_state.dataset.name = session_state.name
                    log_error(f"Dataset name fresh and ready to rumble")
                    return True

            elif not session_state.name:
                log_error(f"Dataset name field is empty!")
                field_placeholder['name'].error(
                    f"Dataset name field is empty!")

                return False

        # ***************************************************topcol3 ***************************************************
        with place['dataset_id'].container():
            # DATASET ID
            st.write(
                f"### __Dataset ID:__ {session_state.dataset.dataset_id}")
            # TITLE
            st.text_input(
                "Dataset Title", value=session_state.dataset.name, key="name", help="Enter the name of the dataset", on_change=check_if_name_exist, args=(place, conn,))
            place['name'] = st.empty()

        # ***************************************************outercol2 ***************************************************
        with place['desc'].container():

            # DESCRIPTION
            desc_col4.write("### **Description:**")

            if session_state.dataset.desc:
                desc = session_state.dataset.desc
            else:
                desc = ""

            st.text_area(
                "Description (Optional)", value=desc, key="desc", help="Enter the description of the dataset")

    with outercol2:
        def append_data_flag_1():
            session_state.append_data_flag = DataPermission.Edit

        def back():
            session_state.append_data_flag = DataPermission.ViewOnly
            # reset values of text input widget
            if 'name' in session_state:
                del session_state.name
                del session_state.desc

        def submit_title_desc_changes():

            # Update database
            log_info("UPDATING")
            if check_if_name_exist(place, conn):
                log_info("START UPDATE")
                session_state.temp_name = session_state.name

                session_state.dataset.update_title_desc(
                    session_state.name, session_state.desc)

                session_state.append_data_flag = DataPermission.ViewOnly

                def show_update_success():
                    update_success_place = outercol2.empty()
                    update_success_place.success(
                        f"Successfully updated fields")
                    log_info("Inside thread")
                    sleep(0.5)
                    log_info("End Sleep in thread")
                    update_success_place.empty()
                update_success_msg_thread = Thread(target=show_update_success)
                add_report_ctx(update_success_msg_thread)
                update_success_msg_thread.start()
                # show_update_success()
                log_info("After thread start")

                # reset values of text input widget
                if 'name' in session_state:
                    del session_state.name
                    del session_state.desc
                if 'dataset_sel' in session_state:
                    del session_state.dataset_sel
                    log_info("DELETE SELECTBOX")
                update_success_msg_thread.join()
            else:
                log_info("NO UPDATE")

        place["delete"] = st.empty()
        # place["delete"].button("Edit dataset", key="edit_dataset2")
        # place['cancel_delete'] = st.empty()

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

        data_selection = data_table(
            session_state.dataset.data_name_list, dataset_columns, key="data_table")

        with st.expander("Append dataset", expanded=False):
            data_uploader(session_state.dataset)
        st.write(f"Selection")
        st.write(data_selection)

        # >>>>>>>>>>>>> DELETE CALLBACK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
        def delete_file_callback(file_selection_list: str, dataset_path: Path):
            if file_selection_list:
                if not dataset_path:
                    log_error(f"Missing dataset path")
                    return

                for filename in file_selection_list:

                    if filename:
                        filepath = Path(dataset_path) / filename

                        # delete file
                        if delete_file_directory(filepath):

                            # remove from Task from DB
                            if Task.delete_task(filename):

                                # update dataset size in database
                                session_state.dataset.update_dataset_size()
                                log_info(
                                    f"{filename} has been removed from dataset {session_state.dataset.name} and all its associated task, annotations and predictions")

                # Appear 'Delete' button
        if data_selection:
            num_data_selection = len(data_selection)
            with place['delete']:

                delete_state = st.button(
                    f"Delete {num_data_selection} selected")

                # Ask user to confirm deletion of selected data from current dataset
                if delete_state:
                    with st.form(key="confirm _delete_data"):
                        st.error(
                            f"Confirm deletion of {num_data_selection} data from {session_state.dataset.name}")
                        confirm_delete_state = st.form_submit_button(
                            f"Confirm delete", on_click=delete_file_callback, args=(data_selection, session_state.dataset.dataset_path,))
                        st.write(data_selection)
                    # st.button("Cancel", key='cancel_delete')
                    place['cancel_delete'].button(
                        "Cancel", key='cancel_delete')
        # <<<<<<<<<< DELETE CALLBACK <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        if session_state.append_data_flag == DataPermission.ViewOnly:
            place["delete"].button(
                "Edit dataset", key="edit_dataset2", on_click=append_data_flag_1)

        elif (session_state.append_data_flag == DataPermission.Edit) and not data_selection:
            with place["delete"].container():
                st.button(
                    "Back", key="back", on_click=back)

                st.button(
                    "Submit Changes", key="edit_title_desc_submit", on_click=submit_title_desc_changes)

    # Add view image
    # Add Edit callback

    # TODO #18

    # <<<<<<<<<<<<<<<<<<<<<<<<<< LOAD EXISTING DATASET <<<<<<<<<<<<<<<<<<<<<#
    # st.write(dataset_dict)


def main():

    # TODO #47 ADD show() function to load existing dataset and new dataset page
    dataset_page = {
        DatasetPagination.Dashboard: dashboard,
        DatasetPagination.New: new_dataset.show
    }
    if 'dataset_pagination' not in session_state:
        session_state.dataset_pagination = DatasetPagination.Dashboard

    project_page_options = ("Dashboard", "Create New Dataset")

    def dataset_page_navigator():
        session_state.dataset_pagination = project_page_options.index(
            session_state.dataset_page_navigator_radio)

    with st.sidebar.expander("Dataset", expanded=True):
        st.radio("", options=project_page_options,
                 index=session_state.dataset_pagination, on_change=dataset_page_navigator, key="dataset_page_navigator_radio")

    dataset_page[session_state.dataset_pagination]()


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
