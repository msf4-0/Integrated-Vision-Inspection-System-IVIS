"""
Title: Editor (TEST)
Date: 22/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
import sys
from pathlib import Path
from PIL import Image
from base64 import b64encode, decode
from io import BytesIO
from enum import IntEnum
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state
from traitlets.traitlets import default
# DEFINE Web APP page configuration
layout = 'wide'
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>> User-defined Modules >>>>
SRC = Path(__file__).resolve().parents[1]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
TEST_MODULE_PATH = SRC / "test" / "test_page" / "module"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

# if str(TEST_MODULE_PATH) not in sys.path:
#     sys.path.insert(0, str(TEST_MODULE_PATH))
# else:
#     pass

from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
from data_manager.database_manager import init_connection
from data_manager.annotation_type_select import annotation_sel
from frontend.editor_manager import BaseEditor, Editor
from core.utils.code_generator import get_random_string
# <<<< User-defined Modules <<<<

# TODO: not used
from frontend.streamlit_labelstudio import st_labelstudio


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


@st.cache
def data_url_encoder():
    """Load Image and generate Data URL in base64 bytes

    Args:
        image (bytes-like): BytesIO object

    Returns:
        bytes: UTF-8 encoded base64 bytes
    """
    chdir_root()  # ./image_labelling
    sample_image = "resources/sample.jpg"
    with Image.open(sample_image) as img:
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='jpeg')

    bb = img_byte_arr.getvalue()
    b64code = b64encode(bb).decode('utf-8')
    data_url = 'data:image/jpeg;base64,' + b64code
    # data_url = f'data:image/jpeg;base64,{b64code}'
    # st.write(f"\"{data_url}\"")

    return data_url, bb


data_url, bb = data_url_encoder()


def get_image_size(image_path):
    """get dimension of image

    Args:
        image_path (str): path to image or byte_like object

    Returns:
        tuple: original_width and original_height
    """
    with Image.open(image_path) as img:
        original_width, original_height = img.size
    return original_width, original_height


def main():
    # >>>> Template >>>>
    chdir_root()  # change to root directory
    # initialise connection to Database
    conn = init_connection(**st.secrets["postgres"])
    with st.sidebar.beta_container():
        st.image("resources/MSF-logo.gif", use_column_width=True)
    # with st.beta_container():
        st.title("Integrated Vision Inspection System", anchor='title')
        st.header(
            "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
        st.markdown("""___""")
    # <<<< Template <<<<

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

    # >>>> EDITOR >>>>>

    # **** Instantiate Editor ****
    if "editor" not in session_state:
        session_state.editor = Editor(get_random_string(8))

    v = annotation_sel()
    col1, col2 = st.beta_columns(2)
    if None not in v:
        (annotationType, annotationConfig_template) = v

        config = annotationConfig_template['config']

        with col1:
            # >>>> Load XML -> Document object
            session_state.editor.xml_doc = session_state.editor.load_xml(config)

            # >>>> get labels

            session_state.editor.childNodes = session_state.editor.get_child(
                'RectangleLabels', 'Label')

            session_state.editor.labels = session_state.editor.get_labels(
                session_state.editor.childNodes)
            tagName_attributes = session_state.editor.get_tagname_attributes(
                session_state.editor.childNodes)
            label = st.text_input('Labels Input', key='labels_input')
            st.info(
                '''Please enter desired labels and choose the labels to be used from the multi-select widget below''')
            labels_chosen = ['Hello', 'World', 'Bye']
            st.multiselect('Labels', options=[
                            'Hello', 'Bye', 'World'], default=labels_chosen, key='labels_select')

            st.write(session_state.editor.xml_doc)
            st.write(session_state.editor.childNodes)
            st.write(session_state.editor.labels)
            st.write(tagName_attributes)
        with st.beta_expander('Editor Config', expanded=False):
            st.code(session_state.editor.to_xml_string(
                pretty=True), language='xml')

        with col2:
            st_labelstudio(config, interfaces, user, task, key='editor_test')


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
