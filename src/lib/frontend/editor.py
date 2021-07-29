"""
Title: Editor
Date: 15/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
import sys
from pathlib import Path
from numpy.lib.function_base import delete
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as SessionState
from PIL import Image
from base64 import b64encode, decode
from io import BytesIO
# DEFINE Web APP page configuration
layout = 'wide'
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>> User-defined Modules >>>>
SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass


from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
from data_manager.database_manager import init_connection
from data_manager.annotation_type_select import annotation_sel
from annotation.annotation_manager import data_url_encoder, load_sample_image, get_image_size
from tasks.results import DetectionBBOX, ImgClassification, SemanticPolygon, SemanticMask
from annotation.annotation_manager import submit_annotations, update_annotations, skip_task, delete_annotation
from enum import IntEnum
# <<<< User-defined Modules <<<<
conn = init_connection(**st.secrets["postgres"])

# NOTE: not used
from frontend.streamlit_labelstudio import st_labelstudio


class EditorFlag(IntEnum):
    SUBMIT = 1
    UPDATE = 2
    DELETE = 3
    SKIP = 4

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
        # st.write(uploaded_files_multi)
        st.subheader(f'Filename: {image_sel}')
        # st.write(image_sel)
        # st.image(uploaded_files_multi[image_name[image_sel]])

        # st.write(uploaded_files_multi[image_name[image_sel]])
        # for image in uploaded_files_multi:
        #     st.write(image)
        #     st.subheader(f"Filename: {image.name}")
        # st.image(uploaded_files_multi[0])
        # st.image(uploaded_files_multi[1])
        # st.image(image)
        data_url = data_url_encoder(
            uploaded_files_multi[image_name[image_sel]])
        original_width, original_height = get_image_size(
            uploaded_files_multi[image_name[image_sel]])
        st.write(
            f"original_width: {original_width}{'|':^8}original_height: {original_height}")

    else:
        data_url = load_sample_image()

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
        "skip"
    ],

    user = {
        'pk': 1,
        'firstName': "Zhen Hao",
        'lastName': "Chu"
    },

    task = {
        "annotations":
            [{
                "id": "1001",
                "lead_time": 15.053,
                "result": [
                    {
                        "original_width": 2242,
                        "original_height": 2802,
                        "image_rotation": 0,
                        "value": {
                            "x": 20,
                            "y": 38.72113676731794,
                            "width": 33.6,
                            "height": 38.18827708703375,
                            "rotation": 0,
                            "rectanglelabels": [
                                "Hello"
                            ]
                        },
                        "id": "Dx_aB91ISN",
                        "from_name": "tag",
                        "to_name": "img",
                        "type": "rectanglelabels"
                    },
                    {
                        "original_width": 2242,
                        "original_height": 2802,
                        "image_rotation": 0,
                        "value": {
                            "x": 48.93333333333334,
                            "y": 25.22202486678508,
                            "width": 15.466666666666667,
                            "height": 16.163410301953817,
                            "rotation": 0,
                            "rectanglelabels": [
                                "Hello"
                            ]
                        },
                        "id": "Dx_a1ISN",
                        "from_name": "tag",
                        "to_name": "img",
                        "type": "rectanglelabels"
                    },
                    {
                        "original_width": 2242,
                        "original_height": 2802,
                        "image_rotation": 0,
                        "value": {
                            "x": 63.866666666666674,
                            "y": 40.49733570159858,
                            "width": 14.000000000000002,
                            "height": 18.29484902309059,
                            "rotation": 0,
                            "rectanglelabels": [
                                "Hello"
                            ]
                        },
                        "id": "Dx_aB9",
                        "from_name": "tag",
                        "to_name": "img",
                        "type": "rectanglelabels"
                    }]}],
        'predictions': [
                {
                    "model_version": "model 1",
                    "created_ago": "3 hours",
                    "result": [
                        {
                            "from_name": "tag",
                            "id": "1",
                            "source": "$image",
                            "to_name": "img",
                            "type": "rectanglelabels",
                            "value": {
                                "height": 11.612284069097889,
                                "rectanglelabels": [
                                    "Hello"
                                ],
                                "rotation": 0,
                                "width": 39.6,
                                "x": 13.2,
                                "y": 34.702495201535505
                            }
                        }
                    ]
                },
                {
                    "model_version": "model 2",
                    "created_ago": "4 hours",
                    "result": [
                        {
                            "from_name": "tag",
                            "id": "t5sp3TyXPo",
                            "source": "$image",
                            "to_name": "img",
                            "type": "rectanglelabels",
                            "value": {
                                "height": 33.61228406909789,
                                "rectanglelabels": [
                                    "World"
                                ],
                                "rotation": 0,
                                "width": 39.6,
                                "x": 13.2,
                                "y": 54.702495201535505
                            }
                        }
                    ]
                }
            ],
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
            results, flag = ImgClassification(
                config, user, task, interfaces, key='img_classification')
        elif annotationType == "Object Detection with Bounding Boxes":
            results = DetectionBBOX(
                config, user, task, interfaces)
        elif annotationType == "Semantic Segmentation with Polygons":
            results, flag = SemanticPolygon(
                config, user, task, original_width, original_height, interfaces)

        elif annotationType == "Semantic Segmentation with Masks":
            results = SemanticMask(
                config, user, task, original_width, original_height, interfaces)
        else:
            pass

# TODO: add required fields
        # if flag and results is not None:
        #     if flag == EditorFlag.submit:
        #         annotation_id = submit_annotations(
        #             results, project_id, users_id, task_id, annotation_id, is_labelled, conn)
        #     elif flag == EditorFlag.update:
        #         update_annotation_return = update_annotations(
        #             results, users_id, annotation_id, conn)
        #     elif flag == EditorFlag.delete:
        #         delete_annotation_return = delete_annotation(annotation_id)
        #     elif flag == EditorFlag.skip:
        #         skipped_task_return = skip_task(task_id, skipped)


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
