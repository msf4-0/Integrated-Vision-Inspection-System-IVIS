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
from data_manager.annotation_type_select import annotation_sel
from data_editor.editor_management import Editor,load_sample_image
from project.project_management import Project
from core.utils.code_generator import get_random_string
# <<<< User-defined Modules <<<<

# TODO: not used
from data_editor.streamlit_labelstudio import st_labelstudio

place = {}


class EditorFlag(IntEnum):
    submit = 1
    update = 2
    delete = 3
    skip = 4

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return EditorFlag[s]
        except KeyError:
            raise ValueError()



def main():

    # >>>> Template >>>>
    chdir_root()  # change to root directory
    # initialise connection to Database
    conn = init_connection(**st.secrets["postgres"])
    with st.sidebar.container():
        st.image("resources/MSF-logo.gif", use_column_width=True)
    # with st.container():
        st.title("Integrated Vision Inspection System", anchor='title')
        st.header(
            "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
        st.markdown("""___""")
    # <<<< Template <<<<

    data_url = load_sample_image()

    interfaces = [
        "panel",
        "controls",
        "side-column"

    ],

    user = {
        'pk': 1,
        'firstName': "Zhen Hao",
        'lastName': "Chu"
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
    # *************************************************************************************
    # >>>> EDITOR >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # *************************************************************************************
    
    # Page title
    st.write("# Editor Config")
    st.markdown("___")
    # **** Instantiate Editor ****
    if "editor" not in session_state:
        session_state.project = Project(7)
        session_state.editor = Editor(session_state.project.id)

    # v = annotation_sel()
    col1, col2 = st.columns([1, 2])
    # if None not in v:
    #     (annotationType, annotationConfig_template) = v

    #     config = annotationConfig_template['config']

    with col1:
        st.text_input("Check column", key="column1")

        # >>>> Load XML -> Document object
        session_state.editor.editor_config = session_state.editor.load_raw_xml()
        xml_doc = session_state.editor.load_xml(
            session_state.editor.editor_config)
        session_state.editor.xml_doc = deepcopy(xml_doc)  # DEEPCOPY

        # >>>> get labels

        session_state.editor.childNodes = session_state.editor.get_child(
            'RectangleLabels', 'Label')

        # labels
        session_state.editor.labels = session_state.editor.get_labels(
            session_state.editor.childNodes)

        tagName_attributes = session_state.editor.get_tagname_attributes(
            session_state.editor.childNodes)

        # add label into list
        def add_label(place):
            st.write(session_state.add_label)
            if session_state.add_label and session_state.add_label not in session_state.editor.labels:

                newChild = session_state.editor.create_label(
                    'RectangleLabels', 'Label', 'value', session_state.add_label)
                log_info(f"newChild: {newChild.attributes.items()}")

                log_info(f"New label added {session_state.editor.labels}")

                # session_state.editor.labels.append(
                #     session_state.labels_input)
            elif session_state.add_label in session_state.editor.labels:
                label_exist_msg = f"Label '{session_state.add_label}' already exists in {session_state.editor.labels}"
                log_error(label_exist_msg)
                place["add_label"].error(label_exist_msg)
                sleep(1)

        def update_labels():
            st.write(session_state.labels_select)
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
                        'Label', 'value', removed_label)
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
            # if len(session_state.editor.labels) > len(session_state.labels_select):
            #     # Action:REMOVE
            #     # Find removed label
            #     removed_label =

        # TODO
        # >>>>>>> ADD LABEL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        st.text_input('Add Label', key='add_label',
                      on_change=add_label, args=(place,))
        place["add_label"] = st.empty()
        place["add_label"].info(
            '''Please enter desired labels and choose the labels to be used from the multi-select widget below''')
        # # labels_chosen = ['Hello', 'World', 'Bye']

        # >>>>>>> REMOVE LABEL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        st.multiselect('Labels', options=session_state.editor.labels,
                       default=session_state.editor.labels, key='labels_select', on_change=update_labels)

        st.write("Labels selected:")
        st.write(session_state.labels_select)

        # >>>>>>>>> try to create node

        # st.write("Create Label")
        # newChild = session_state.editor.create_label(
        #     'RectangleLabels', 'Label', 'value', 'aruco')
        # st.write(newChild.attributes.items())
        # st.write("Remove Label")

        # st.write(session_state.editor.remove_labels(
        #     'Label', 'value', 'Hello')[0].attributes.items())
        # # st.write(removeChild)
        # st.write("Edit Label")
        # st.write(session_state.editor.edit_labels(
        #     'Label', 'value', 'aruco', 'Car'))

        # newChild = session_state.editor.create_label(
        #     'RectangleLabels', 'Label', 'value', 'airport')
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
        st.text_input("Check column", key="column2")
        st_labelstudio(config2, interfaces, user, task, key='editor_test')
        st.write("Project Class")
        st.write(vars(session_state.project))


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
