"""
Title: Editor Config
Date: 22/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
import sys
from pathlib import Path
from enum import IntEnum
from copy import deepcopy
from time import sleep
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state
# DEFINE Web APP page configuration
layout = 'wide'
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
TEST_MODULE_PATH = SRC / "test" / "test_page" / "module"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
from data_manager.database_manager import init_connection
from data_editor.editor_management import Editor, EditorFlag, load_sample_image
from data_editor.streamlit_labelstudio import st_labelstudio
from project.project_management import Project
from deployment.deployment_management import DEPLOYMENT_TYPE, DeploymentType
# <<<< User-defined Modules <<<<

# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])
place = {}


def editor_config(project_id: int, deployment_type: str):

    chdir_root()  # change to root directory

    # >>>> LOAD SAMPLE IMAGE >>>>
    data_url = load_sample_image()

# *********************** EDITOR SETUP ****************************************************
    interfaces = [
        "panel",
        "controls",
        "side-column"

    ],

    user = {
        'pk': 1,
        'firstName': "John",
        'lastName': "Snow"
    },

    task = {
        "annotations":
            [],
        'predictions': [],
        'id': 1,
        'data': {
            # 'image': "https://app.heartex.ai/static/samples/sample.jpg"
            'image': f'{data_url}'
            }
    }
    # *********************** EDITOR SETUP ****************************************************

# *************************************************************************************
# >>>> EDITOR >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# *************************************************************************************

# ******** SESSION STATE *********************************************************

    if "editor" not in session_state:
        # TODO Pass current project ID into here
        # session_state.project = Project(7)

        session_state.editor = Editor(project_id, deployment_type)

    # ******** SESSION STATE *********************************************************


# ******************************************START*******************************************
    # Page title
    st.write("# Editor Config")
    st.markdown("___")

    # ************COLUMN PLACEHOLDERS *****************************************************

    # >>>> MAIN COLUMNS
    col1, col2 = st.columns([1, 2])

    # >>>> Add 'save' button
    save_col1, save_col2 = st.columns([1, 2])

    # >>>> To display variables during dev
    lowercol1, lowercol2 = st.columns([1, 2])

    with col1:

        # >>>> LOAD LABELS
        session_state.editor.labels = session_state.editor.get_labels()
        if 'labels_select' not in session_state:
            session_state.labels_select = session_state.editor.labels

        # TODO remove
        tagName_attributes = session_state.editor.get_tagname_attributes(
            session_state.editor.childNodes)

        def add_label(place):

            if session_state.add_label and session_state.add_label not in session_state.editor.labels:

                newChild = session_state.editor.create_label(
                    'value', session_state.add_label)

                log_info(f"newChild: {newChild.attributes.items()}")

                log_info(f"New label added {session_state.editor.labels}")
                session_state.editor.labels = session_state.editor.get_labels()
                if 'labels_select' in session_state:
                    del session_state.labels_select

            elif session_state.add_label in session_state.editor.labels:
                label_exist_msg = f"Label '{session_state.add_label}' already exists in {session_state.editor.labels}"
                log_error(label_exist_msg)
                place["add_label"].error(label_exist_msg)
                sleep(1)
                place["add_label"].empty()

        def update_labels():

            diff_12 = set(session_state.editor.labels).difference(
                session_state.labels_select)  # set 1 - set 2 REMOVAL
            diff_21 = set(session_state.labels_select).difference(
                session_state.editor.labels)  # set 2 - set 1 ADDITION
            if diff_12:
                log_info("Removal")
                removed_label = list(diff_12).pop()
                try:
                    session_state.editor.labels.remove(
                        removed_label)
                    session_state.editor.labels = sorted(
                        session_state.editor.labels)
                    # TODO: function to remove DOM
                    removedChild = session_state.editor.remove_label(
                        'value', removed_label)
                    log_info(f"removedChild: {removedChild}")
                    log_info(f"Label removed {session_state.editor.labels}")

                except ValueError as e:
                    st.error(f"{e}: Label already removed")
            elif diff_21:
                print("Addition")
                # added_label = list(diff_21).pop()
                # session_state.editor.labels.append(added_label)
                session_state.editor.labels = sorted(
                    session_state.editor.labels + list(diff_21))
                # TODO: function to add DOM

            else:
                print("No Change")
                pass
            session_state.editor.labels = session_state.editor.get_labels()
            if 'labels_select' in session_state:
                del session_state.labels_select

            # if len(session_state.editor.labels) > len(session_state.labels_select):
            #     # Action:REMOVE
            #     # Find removed label
            #     removed_label =

        # >>>>>>> ADD LABEL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        st.text_input('Add Label', key='add_label',
                      on_change=add_label, args=(place,))
        place["add_label"] = st.empty()
        place["add_label"].info(
            '''Please enter desired labels and choose the labels to be used from the multi-select widget below''')
        # # labels_chosen = ['Hello', 'World', 'Bye']
        log_info(f"Before multi {session_state.editor.labels}")
        # >>>>>>> REMOVE LABEL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        st.multiselect('Labels', options=session_state.editor.labels,
                       key='labels_select', on_change=update_labels)

    # TODO ADD 'SAVE' BUTTON
    with save_col1:
        def save_editor_config(deployment_type):
            log_info("Updating Editor Config......")

            if session_state.editor.update_editor_config(deployment_type):

                #>>>> Display success message
                update_success_place = st.empty()
                update_success_place.success(
                    f"Successfully updated editor configurations")
                sleep(0.5)
                update_success_place.empty()

                # if 'editor' in session_state:
                #     del session_state.editor

        st.button('Save', key='save_editor_config',
                  on_click=save_editor_config, args=(deployment_type,))

        # >>>>>>>>>> TODO #66 Add Color picker for Bbox, Segmentation Polygons and Segmentation Masks >>>>>>>>>>>>>>
    with lowercol1:
        st.write("Labels selected:")
        st.write(session_state.labels_select)
        st.write("Doc")
        st.write(session_state.editor.xml_doc)
        st.write("Label Childnodes")
        st.write(session_state.editor.childNodes)
        st.write("Labels")
        st.write(session_state.editor.labels)
        st.write("Attributes")
        st.write(tagName_attributes)
        st.write("Editor Class")
        st.write(vars(session_state.editor))

    with st.expander('Editor Config', expanded=True):
        config2 = session_state.editor.to_xml_string(
            pretty=True)
        st.code(config2, language='xml')

    with col2:
        # st.text_input("Check column", key="column2")
        st_labelstudio(config2, interfaces, user, task, key='editor_test')


def main():
    RELEASE = False

    # ****************** TEST ******************************
    if not RELEASE:

        # ************************TO REMOVE************************
        with st.sidebar.container():
            st.image("resources/MSF-logo.gif", use_column_width=True)
            st.title("Integrated Vision Inspection System", anchor='title')
            st.header(
                "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
            st.markdown("""___""")
        # ************************TO REMOVE************************
        project_id = 7
        deployment_type = DEPLOYMENT_TYPE["Object Detection with Bounding Boxes"] # get enum ->2
        editor_config(project_id, deployment_type)


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
