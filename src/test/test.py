"""
TEST
"""

import sys
from pathlib import Path
from time import sleep
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state

# DEFINE Web APP page configuration
layout = 'wide'
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[1]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
# TEST_MODULE_PATH = SRC / "test" / "test_page" / "module"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
from copy import deepcopy
from core.webcam import webcam_webrtc
from core.utils.helper import check_filetype
from data_manager.database_manager import init_connection
from data_manager.dataset_management import Dataset, query_dataset_list, get_dataset_name_list

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration <<<<
conn = init_connection(**st.secrets["postgres"])
# <<<< Variable Declaration <<<<


def test():
    log_info("At top")
    if 'place' not in session_state:
        session_state.place = {}
    existing_dataset, _ = query_dataset_list()
    dataset_dict = get_dataset_name_list(existing_dataset)

    # if 'dataset' not in session_state:
    #     session_state.dataset = Dataset(dataset_dict['My Third Dataset'])
    if 'append_data_flag' not in session_state:
        session_state.append_data_flag = 0
    if 'selectbox_flag' not in session_state:
        session_state.selectbox_flag = 0
    # if 'dataset_sel' in session_state:
    #     del session_state.dataset_sel
    st.write(session_state.append_data_flag)

    # ************COLUMN PLACEHOLDERS *****************************************************
    # >>>> Column for Create New button and Dataset selectbox
    topcol1, _, topcol2, _, top_col3 = st.columns(
        [0.15, 0.3, 2, 0.1, 3.5])
    # topcol3 for dataset ID
    session_state.place["dataset_id"] = top_col3.empty()
    # >>>> COLUMNS for BUTTONS
    _, button_col1, _, button_col2, _, button_col3, _, desc_col4 = st.columns(
        [0.115, 0.375, 0.3, 0.75, 0.3, 0.375, 0.1, 3.15])

    # >>>> Main placeholders
    outercol1, _, outercol2 = st.columns(
        [2.5, 0.1, 3.5])
    session_state.place["desc"] = outercol2.empty()
    # column placeholder for variable/class objects
    _, varscol1, varscol2, varscol3 = st.columns([0.4, 2, 3.5, 0.15])
    # ************COLUMN PLACEHOLDERS *****************************************************
    with topcol1:
        st.write("## ")
        st.button("➕️", key="new_dataset", help="Create new dataset")

    # TODO: removed 'dataset selection'????
    with topcol2:
        def append_data_flag_0():
            session_state.append_data_flag = 0
            session_state.selectbox_flag = 1
            if "dataset" not in session_state:
                session_state.dataset = Dataset(
                    dataset_dict[session_state.dataset_sel])
            else:
                st.write(session_state.dataset_sel)

                session_state.dataset = Dataset(
                    dataset_dict.get(session_state.dataset_sel))

            if 'name' in session_state:
                del session_state.name
                del session_state.desc
        log_info("In selectbox")
        # st.write(dataset_dict)

        try:
            dataset_selection = st.selectbox("Select Dataset", options=dataset_dict,
                                             key="dataset_sel", on_change=append_data_flag_0, help="Select dataset to view details")
        except:
            existing_dataset, _ = query_dataset_list()
            dataset_dict = get_dataset_name_list(existing_dataset)
            dataset_selection = st.selectbox("Select Dataset", options=dataset_dict,
                                             key="dataset_sel", on_change=append_data_flag_0, help="Select dataset to view details")
        st.write(dataset_selection)
        # def instantiate_dataset_class(dataset_dict):
    if session_state.selectbox_flag == 0:
        if 'temp_name' not in session_state:
            session_state.temp_name = None
        if "dataset" not in session_state:
            session_state.dataset = Dataset(
                dataset_dict[session_state.dataset_sel])
        else:
            st.write(session_state.dataset_sel)

            session_state.dataset = Dataset(
                dataset_dict.get(session_state.dataset_sel))
            # except:
            #     session_state.dataset = Dataset(
            #         dataset_dict.get(session_state.temp_name))

            log_info("At session state for dataset")
            # st.write(session_state.temp_name)
    # ******************************************************************************
    log_info("Outercol2 Zone")
    if session_state.append_data_flag == 0:
        with session_state.place['dataset_id'].container():
            # DATASET ID
            st.write(
                f"### __Dataset ID:__ {session_state.dataset.dataset_id}")
            # TITLE
            st.write(f"# {session_state.dataset.name}")

        with session_state.place['desc'].container():

            # DESCRIPTION
            desc_col4.write("### **Description:**")
            if session_state.dataset.desc:
                st.write(f"{session_state.dataset.desc}")
            else:
                st.write(f"No description")

            # TODO Add edit button -- CALLBACK?
            # st.button("Edit dataset", key="edit_dataset")
            # TODO Add delete button
    elif session_state.append_data_flag == 1:
        def check_if_name_exist(field_placeholder, conn):
            context = ['name', session_state.name]

            # Do not check if name same as current dataset name
            if (session_state.name) and (session_state.name != session_state.dataset.name):
                if session_state.dataset.check_if_exist(context, conn):
                    field_placeholder['name'].error(
                        f"Dataset name used. Please enter a new name")
                    sleep(1)
                    log_error(f"Dataset name used. Please enter a new name")
                    return False
                else:
                    session_state.dataset.name = session_state.name
                    log_error(f"Dataset name fresh and ready to rumble")
                    return True
            elif not session_state.name:
                log_error(f"Dataset name field is empty!")
                field_placeholder['name'].error(
                    f"Dataset name field is empty!")
                sleep(1)
                return False

        with session_state.place['dataset_id'].container():
            # DATASET ID
            st.write(
                f"### __Dataset ID:__ {session_state.dataset.dataset_id}")
            # TITLE
            st.text_input(
                "Dataset Title", value=session_state.dataset.name, key="name", help="Enter the name of the dataset", on_change=check_if_name_exist, args=(session_state.place, conn,))
            session_state.place['name'] = st.empty()
            st.write(session_state.name)
        with session_state.place['desc'].container():

            # DESCRIPTION
            desc_col4.write("### **Description:**")
            if session_state.dataset.desc:
                desc = session_state.dataset.desc
            else:
                desc = ""
            st.text_area(
                "Description (Optional)", value=desc, key="desc", help="Enter the description of the dataset")
            st.write(session_state.desc)
            # TODO Add edit button -- CALLBACK?
            # st.button("Edit dataset", key="edit_dataset")
            # TODO Add delete button

    with outercol2:
        def append_data_flag_1():
            session_state.append_data_flag = 1

        def back():
            session_state.append_data_flag = 0
            # reset values of text input widget
            if 'name' in session_state:
                del session_state.name
                del session_state.desc

        def submit_title_desc_changes():

            # Update database

            if check_if_name_exist(session_state.place, conn):
                session_state.temp_name = session_state.dataset.name
                session_state.dataset.update_title_desc(
                    session_state.name, session_state.desc)
                session_state.append_data_flag = 0
                st.success(f"Successfully updated fields")
                sleep(1)
                # reset values of text input widget
                if 'name' in session_state:
                    del session_state.name
                    del session_state.desc

        # ****TEMP*******
        data_selection = 0

        session_state.place["delete"] = st.empty()
        if session_state.append_data_flag == 0:
            session_state.place["delete"].button(
                "Edit dataset", key="edit_dataset2", on_click=append_data_flag_1)
        elif (session_state.append_data_flag == 1) and not data_selection:
            with session_state.place["delete"].container():
                st.button(
                    "Back", key="back", on_click=back)

                st.button(
                    "Submit Changes", key="edit_title_desc_submit", on_click=submit_title_desc_changes)

        session_state.place['cancel_delete'] = st.empty()


def main():
    TEST_FLAG = True

    # ****************** TEST ******************************
    if TEST_FLAG:

        chdir_root()  # change to root directory
        # initialise connection to Database

        test()


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
