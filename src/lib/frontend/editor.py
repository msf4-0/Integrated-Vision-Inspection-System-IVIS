"""
Title: Editor
Date: 15/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
import sys
from pathlib import Path
import psycopg2
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as SessionState
from PIL import Image
from base64 import b64encode, decode

# DEFINE Web APP page configuration
layout = 'wide'
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>> User-defined Modules >>>>
SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
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
from tasks.results import DetectionBBOX, ImgClassification, SemanticPolygon, SemanticMask
# <<<< User-defined Modules <<<<

# TODO: not used
from frontend.streamlit_labelstudio import st_labelstudio


@st.cache
def data_url_encoder(image):
    """Load Image and generate Data URL in base64 bytes

    Args:
        image (bytes-like): BytesIO object

    Returns:
        bytes: UTF-8 encoded base64 bytes
    """

    bb = image.read()
    b64code = b64encode(bb).decode('utf-8')
    data_url = 'data:' + image.type + ';base64,' + b64code
    # data_url = f'data:image/jpeg;base64,{b64code}'
    # st.write(f"\"{data_url}\"")

    return data_url


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

    with st.sidebar.beta_container():
        st.markdown("""
        # Batch Image Upload """)

        # streamlit.file_uploader(label, type=None, accept_multiple_files=False, key=None, help=None)
        uploaded_files_multi = st.file_uploader(
            label="Upload Image", type=['jpg', "png", "jpeg"], accept_multiple_files=True, key="upload")
        # if uploaded_files_multi is not None:
        # image_multi = Image.open(uploaded_files)
        # st.image(image_multi, caption="Uploaded Image")

    st.markdown("""
        # Batch Image Upload """)
    if uploaded_files_multi:
        image_name = {}
        image_list = []
        i = 0
        st.write(uploaded_files_multi[0].type)
        for image in uploaded_files_multi:
            image_name[image.name] = i
            image_list.append(image.name)
            i += 1
        image_sel = st.sidebar.selectbox(
            "Select image", image_list)
        # with st.beta_expander('Show image'):
        st.write(uploaded_files_multi)
        st.subheader(f'Filename: {image_sel}')
        st.write(image_sel)
        # st.image(uploaded_files_multi[image_name[image_sel]])

        st.write(uploaded_files_multi[image_name[image_sel]])
        # for image in uploaded_files_multi:
        #     st.write(image)
        #     st.subheader(f"Filename: {image.name}")
        # st.image(uploaded_files_multi[0])
        # st.image(uploaded_files_multi[1])
        # st.image(image)
        data_url = data_url_encoder(uploaded_files_multi[image_name[image_sel]])
        original_width, original_height = get_image_size(
            uploaded_files_multi[image_name[image_sel]])
        st.write(
            f"original_width: {original_width}{'|':^8}original_height: {original_height}")

    else:
        data_url = "https://htx-misc.s3.amazonaws.com/opensource/label-studio/examples/images/nick-owuor-astro-nic-visuals-wDifg5xc9Z4-unsplash.jpg"

    pass

    interfaces = [
        "panel",
        "update",
        "submit",
        "controls",
        "side-column",
        "annotations:menu",
        "annotations:add-new",
        "annotations:delete",
        "predictions:menu",
    ],

    user = {
        'pk': 1,
        'firstName': "Zhen Hao",
        'lastName': "Chu"
    },

    task = {
        'annotations': [],
        'predictions': [],
        'id': 1,
        'data': {
            # 'image': "https://htx-misc.s3.amazonaws.com/opensource/label-studio/examples/images/nick-owuor-astro-nic-visuals-wDifg5xc9Z4-unsplash.jpg"
            'image': f'{data_url}'
        }
    }

    v = annotation_sel()
    if None not in v:
        (annotationType, annotationConfig_template) = v

        config = annotationConfig_template['config']
        if annotationType == "Image Classification":
            results = ImgClassification(
                config, user, task, interfaces, key='img_classification')
        elif annotationType == "Object Detection with Bounding Boxes":
            results = DetectionBBOX(
                config, user, task,interfaces)
        elif annotationType == "Semantic Segmentation with Polygons":
            results = SemanticPolygon(
                config, user, task, original_width, original_height, interfaces)

        elif annotationType == "Semantic Segmentation with Masks":
            results = SemanticMask(
                config, user, task, original_width, original_height, interfaces)
        else:
            pass


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
