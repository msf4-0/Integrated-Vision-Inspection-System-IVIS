"""
Title: Editor
Date: 15/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
import sys
from pathlib import Path
from enum import IntEnum
from PIL import Image
from base64 import b64encode, decode
from io import BytesIO
from threading import Thread
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state
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
from core.utils.helper import create_dataframe, check_if_exists
from project.project_management import Project
from frontend.editor_manager import Editor
from user.user_management import User
from data_manager.database_manager import init_connection
from annotation.annotation_manager import Annotations, NewAnnotations, NewTask, Task
from tasks.results import DetectionBBOX, ImgClassification, SemanticPolygon, SemanticMask

# <<<< User-defined Modules <<<<
conn = init_connection(**st.secrets["postgres"])

# NOTE: not used
from frontend.streamlit_labelstudio import st_labelstudio
from streamlit.report_thread import add_report_ctx


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


EDITOR_CONFIG = {"Image Classification": ImgClassification, "Object Detection with Bounding Boxes": DetectionBBOX,
                 "Semantic Segmentation with Polygons": SemanticPolygon, "Semantic Segmentation with Masks": SemanticMask}


def show():

    chdir_root()  # change to root directory

    with st.sidebar.beta_container():

        st.image("resources/MSF-logo.gif", use_column_width=True)
    # with st.beta_container():
        st.title("Integrated Vision Inspection System", anchor='title')

        st.header(
            "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
        st.markdown("""___""")

    # ******** SESSION STATE ***********************************************************
    # TODO
    # if "current_page" not in session_state:  # KIV
    #     session_state.current_page = "All Trainings"
    #     session_state.previous_page = "All Trainings"

    if "project" not in session_state:
        # TODO: query all project
        session_state.project = Project(7)
        session_state.editor = Editor(session_state.project.id)
        session_state.user = User(1)
        session_state.annotation = None
        session_state.new_annotation = None
        session_state.task = None

        # set random project ID before getting actual from Database
    session_state.project.query_all_fields()
    # ******** SESSION STATE *********************************************************

    # >>>> TRAINING SIDEBAR >>>>
    # training_page_options = ("All Trainings", "New Training")
    # with st.sidebar.beta_expander("Training Page", expanded=True):
    #     session_state.current_page = st.radio("", options=training_page_options,
    #                                           index=0)
    # <<<< TRAINING SIDEBAR <<<<

    # Page title
    st.write(f'# Project Name: {session_state.project.name}')
    st.write("## **Image Labelling**")
    dt_place, project_id_place = st.beta_columns([3, 1])
    with dt_place:
        st.write("### __Deployment Type:__",
                 f"{session_state.project.deployment_type}")
    with project_id_place:
        st.write(f"### **Project ID:** {session_state.project.id}")
    st.markdown("___")

    # get dataset name list
    session_state.project.datasets = session_state.project.query_project_dataset_list()
    session_state.project.dataset_name_list, session_state.project.dataset_name_id = session_state.project.get_dataset_name_list()
    # load_dataset = Thread(target=session_state.project.load_dataset())
    # add_report_ctx(load_dataset)
    # load_dataset.start()
    # load_dataset.join()

    session_state.project.dataset_list = session_state.project.load_dataset()
    # print(session_state.project.datasets)
    # st.image( session_state.project.dataset_list['My Third Dataset']["IMG_20210315_184229.jpg"],channels='BGR')
# **************************DATA SELECTOR ********************************************
    # _, col1, _, col2, _, col3, _ = st.beta_columns(
    #     [0.2, 1, 0.2, 1, 0.2, 1, 0.2])

    col1, col2 = st.beta_columns([1, 2])
    dataset_selection = col1.selectbox(
        "Dataset", options=session_state.project.dataset_name_list, key="dataset_sel")

    with col1.beta_container():
        project_id = session_state.project.id
        dataset_id = session_state.project.dataset_name_id[dataset_selection]

        # ************************* CALLBACK FUNCTION ************************************
        # >>>> Check if Task exists in 'Task' table >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        def check_if_task_exist(project_id, dataset_id, conn):
            data = session_state.project.dataset_list[dataset_selection][session_state.data_sel]
            if Task.check_if_task_exists(session_state.data_sel, project_id, dataset_id, conn):

                # NOTE: LOAD TASK
                # if 'task' not in session_state:
                session_state.task = Task(data,
                                          session_state.data_sel, project_id, dataset_id)
                log_info(
                    f"Task exists for Task ID: {session_state.task.id} for {session_state.task.name}")

                # >>>> Check if annotations exists
                if Annotations.check_if_annotation_exists(session_state.task.id, project_id, conn):

                    # NOTE: LOAD ANNOTATIONS
                    # if 'annotation' not in session_state:
                    session_state.annotation = Annotations(
                        session_state.task)
                    log_info(
                        f"Annotation {session_state.annotation.id} exists for Task ID: {session_state.task.id} for {session_state.task.name}")
                else:
                    # NOTE: LOAD TASK
                    session_state.annotation = NewAnnotations(
                        session_state.task)
                    log_info(
                        f"Annotation does not exist for Task ID: {session_state.task.id} for {session_state.task.name}")
                    pass

            else:
                # NOTE: CREATE TASK
                # Insert as new task entry if not exists
                task_id = NewTask.insert_new_task(
                    session_state.data_sel, project_id, dataset_id)
# NOTE                # if 'task' not in session_state:
                # Instantiate task as 'Task' Class object
                session_state.task = Task(data,
                                          session_state.data_sel, project_id, dataset_id)
                session_state.annotation = NewAnnotations(
                    session_state.task)
                log_info(
                    f"Created New Task for ID {session_state.task.id} for {session_state.task.name}")

        # >>>> Check if Task exists in 'Task' table >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # ******************* Generate List of Datas ***************************************
        try:
            data_list = sorted(
                [k for k, v in session_state.project.dataset_list[dataset_selection].items()])
        except ValueError as e:
            log_error(
                f"{e}: Dataset Loading error causing list to be non iterable")
        # else:
        #     data_list = []

        st.selectbox(
            "Data", options=data_list, key="data_sel")
        st.button(
            "Confirm", key='data_sel_button', on_click=check_if_task_exist, args=(project_id, dataset_id, conn,))


# *************************EDITOR**********************************************

    # interfaces = [
    #     "panel",
    #     "update",
    #     "submit",
    #     "controls",
    #     "side-column",
    #     "annotations:menu",
    #     "annotations:add-new",
    #     "annotations:delete",
    #     "predictions:menu",
    #     "skip"
    # ],
    @st.cache
    def load_sample_image():
        """Load Image and generate Data URL in base64 bytes

        Args:
            image (bytes-like): BytesIO object

        Returns:
            bytes: UTF-8 encoded base64 bytes
        """
        chdir_root()  # ./image_labelling
        log_info("Loading Sample Image")
        sample_image = "resources/sample.jpg"
        with Image.open(sample_image) as img:
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='jpeg')

        bb = img_byte_arr.getvalue()
        b64code = b64encode(bb).decode('utf-8')
        data_url = 'data:image/jpeg;base64,' + b64code
        # data_url = f'data:image/jpeg;base64,{b64code}'
        # st.write(f"\"{data_url}\"")

        return data_url

    

    user = {
        'pk': session_state.user.id,
        'firstName': session_state.user.first_name,
        'lastName': session_state.user.last_name
    },
    try:
        annotations_dict = session_state.annotation.generate_annotation_dict()
        task = session_state.task.generate_editor_format(
            annotations_dict=annotations_dict, predictions_dict=None)
        
        
    except Exception as e:
        log_error(
            f"{e}: No data selected. Could not generate editor format based on task")
        task = {
                "annotations":
                    [],
                'predictions': [],
                'id': 1,
                'data': {
                    # 'image': "https://app.heartex.ai/static/samples/sample.jpg"
                    'image': None
                        }
            }
    # st.json(task)


    # task = {
    #     "annotations":
    #         [],
    #     'predictions': [],
    #     'id': 1,
    #     'data': {
    #         # 'image': "https://app.heartex.ai/static/samples/sample.jpg"
    #         'image': load_sample_image()
    #             }
    # }


    st.json(user)
    st.write(f"{session_state.project.deployment_type}")
    if session_state.editor.editor_config:
        with col2:
            results = EDITOR_CONFIG[session_state.project.deployment_type](
                session_state.editor.editor_config, user, task)
            st.write(f"Results:{results}")

# NOTE: Load ^ into results.py

# *************************EDITOR**********************************************

    col1, col2 = st.beta_columns([1, 2])
    col1.text_input("Check column", key="column1")
    col2.text_input("Check column", key="column2")

    col1, col2, col3 = st.beta_columns(3)
    col1.write(vars(session_state.project))
    # col1.write(session_state.project.dataset_list['My Third Dataset'])
    col2.write(vars(session_state.editor))
    # col3.write(vars(session_state.task))
    st.write(vars(session_state.user))


def main():
    show()


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
