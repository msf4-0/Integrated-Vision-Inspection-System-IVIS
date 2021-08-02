"""
Title: Editor
Date: 15/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
import sys
from pathlib import Path
from enum import IntEnum
from threading import Thread
from time import sleep
from copy import deepcopy
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
from core.utils.helper import create_dataframe
from project.project_management import Project
from frontend.editor_manager import Editor
from user.user_management import User
from data_manager.database_manager import init_connection
from annotation.annotation_manager import Annotations, NewAnnotations, NewTask, Task, load_buffer_image
from tasks.results import DetectionBBOX, ImgClassification, SemanticPolygon, SemanticMask

# <<<< User-defined Modules <<<<
conn = init_connection(**st.secrets["postgres"])

# NOTE: not used********************************************
from frontend.streamlit_labelstudio import st_labelstudio
from streamlit.report_thread import add_report_ctx


class EditorFlag(IntEnum):
    START = 0
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
    log_info("Start")
    chdir_root()  # change to root directory

    if "data_sel" in session_state:
        del session_state.data_sel

    with st.sidebar.beta_container():

        st.image("resources/MSF-logo.gif", use_column_width=True)
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
    if 'data_list' not in session_state:
        session_state.data_list = {}
    if "labelling_interface" not in session_state:
        session_state.labelling_interface = ([], [], [], 0)
    if "new_annotation_flag" not in session_state:
        session_state.new_annotation_flag = 0
    if "data_selection" not in session_state:
        session_state.data_selection = None

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

    # NOTE: Is this neccesary? Already loaded at Init
    # get dataset name list
    # session_state.project.datasets = session_state.project.query_project_dataset_list()
    # session_state.project.dataset_name_list, session_state.project.dataset_name_id = session_state.project.get_dataset_name_list()

    # TODO: Require threading?

    # load_dataset = Thread(target=session_state.project.load_dataset())
    # add_report_ctx(load_dataset)
    # load_dataset.start()
    # load_dataset.join()

# **************************DATASET SELECTOR ********************************************
    # _, col1, _, col2, _, col3, _ = st.beta_columns(
    #     [0.2, 1, 0.2, 1, 0.2, 1, 0.2])

    col1, col2 = st.beta_columns([1, 1])
    dataset_selection = col1.selectbox(
        "Dataset", options=session_state.project.dataset_name_list, key="dataset_sel")
    project_id = session_state.project.id
    # get dataset_id
    dataset_id = session_state.project.dataset_name_id[dataset_selection]

    # ************************* CALLBACK FUNCTION ************************************
    # >>>> Check if Task exists in 'Task' table >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def check_if_task_exist(project_id, dataset_id, conn):
        log_info("Enter Callback")
        session_state.new_annotation_flag = 0
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
                session_state.annotation = Annotations(
                    session_state.task)
                log_info(
                    f"Annotation does not exist for Task ID: {session_state.task.id} for {session_state.task.name}")
                pass

        else:
            # NOTE: CREATE TASK
            # Insert as new task entry if not exists
            task_id = NewTask.insert_new_task(
                session_state.data_sel, project_id, dataset_id)
# NOTE
            # Instantiate task as 'Task' Class object
            session_state.task = Task(data,
                                      session_state.data_sel, project_id, dataset_id)
            session_state.annotation = Annotations(
                session_state.task)
            log_info(
                f"Created New Task for ID {session_state.task.id} for {session_state.task.name}")

# **************************DATA SELECTOR ********************************************

    with col1.form(key="data_select"):
        # with col1.beta_container():

        # TODO ******************* Generate List of Datas ***************************************
        try:
            # data_list = sorted(
            #     [k for k, v in session_state.project.dataset_list.get(dataset_selection).items()])
            # data_name_list = deepcopy(session_state.project.data_name_list)
            # data_name_list = session_state.project.get_data_name_list()

            data_list = (
                session_state.project.data_name_list.get(dataset_selection))
            log_info("Loading data name list......")

        except ValueError as e:
            log_error(
                f"{e}: Dataset Loading error causing list to be non iterable")

        try:
            session_state.data_selection = st.selectbox(
                "Data", options=data_list, key="data_sel")

        except ValueError as e:
            log_error(f"{e}")

        st.form_submit_button(
            "Confirm", on_click=check_if_task_exist, args=(project_id, dataset_id, conn,))
        # st.write(vars(session_state.annotation))

# >>>>>>>> TABLE OF DATA LIST >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    col1.write("### Table of Data")
    annotation_task_column_names = session_state.project.get_annotation_task_list()

    if "annotation_task_page" not in session_state:
        session_state.annotation_task_page = 0

    def next_page():
        session_state.annotation_task_page += 1

    def prev_page():
        session_state.annotation_task_page -= 1

    # if session_state.project.annotation_task_join:
    with col1:
        start = 10 * session_state.annotation_task_page
        end = start + 10

        df = create_dataframe(session_state.project.annotation_task_join, column_names=annotation_task_column_names,
                              sort=True, sort_by='Dataset Name', asc=False, date_time_format=True)
        df_slice = df.iloc[start:end]
        # if data_selection:
        # TODO #14 Data Selection not updated

        def highlight_row(x, selections):

            if x["Task Name"] in selections:

                return ['background-color: #90a4ae'] * len(x)
            else:
                return ['background-color: '] * len(x)

        styler = df_slice.style.apply(
            highlight_row, selections=session_state.data_selection, axis=1)

        # else:
        #     styler = df_slice.style

        st.table(styler.set_properties(**{'text-align': 'center'}).set_table_styles(
            [dict(selector='th', props=[('text-align', 'center')])]))

    # >>>>>>>> BUTTON >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    _, button_col1, _, button_col2, _, button_col3, _ = st.beta_columns(
        [0.2, 0.15, 0.5, 0.45, 0.5, 0.15, 2])

    num_data_per_page = 10
    num_data_page = len(session_state.project.annotation_task_join
                        ) // num_data_per_page
    # df["Task Name"]
    if num_data_page > 1:
        if session_state.annotation_task_page < num_data_page:
            button_col3.button(">", on_click=next_page)
        else:
            # this makes the empty column show up on mobile
            button_col3.write("")

        if session_state.annotation_task_page > 0:
            button_col1.button("<", on_click=prev_page)
        else:
            # this makes the empty column show up on mobile
            button_col1.write("")

        button_col2.write(
            f"Page {1+session_state.annotation_task_page} of {num_data_page}")
    # >>>>>>>> BUTTON >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>> TABLE OF DATA LIST >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# *************************EDITOR**********************************************

# ***************TEMP**********************
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
# ***************TEMP**********************
    # >>>> User
    user = {
        'pk': session_state.user.id,
        'firstName': session_state.user.first_name,
        'lastName': session_state.user.last_name
    },
    try:
        if session_state.labelling_interface:
            # if there are results
            # if 0, CHANGES LOCKED
            if session_state.labelling_interface[0] and session_state.new_annotation_flag != 0:
                log_info(session_state.new_annotation_flag)
                # CRUD for annotation results
                result, flag = EDITOR_CONFIG.get(
                    session_state.project.deployment_type, DetectionBBOX)(session_state.labelling_interface)  # generate results for annotations
                log_info(f"Flag at main: {flag}")

                if result:
                    if flag == EditorFlag.START:  # LOAD EDITOR
                        log_info("Editor Loaded (In result)")
                        pass

                    elif flag == EditorFlag.SUBMIT:  # NEW ANNOTATION
                        try:

                            session_state.annotation.submit_annotations(
                                result, session_state.user.id, conn)

                            log_info(
                                f"New submission for Task {session_state.task.name} with Annotation ID: {session_state.annotation.id}")
                            # st.json(session_state.annotation.result)
                            # st.experimental_rerun()

                        except Exception as e:
                            log_error(f"{e}: New Annotation error")

                    elif flag == EditorFlag.UPDATE:  # UPDATE ANNOTATION
                        try:

                            session_state.annotation.result = session_state.annotation.update_annotations(
                                result, session_state.user.id, conn).result

                            # log_info(
                            #     f"Update annotations for Task {session_state.task.name} with Annotation ID: {session_state.annotation.id}")
                            # st.json(session_state.annotation.result)
                            # st.experimental_rerun()
                        except Exception as e:
                            log_error(f"{e}: Update annotation error")

                    elif flag == EditorFlag.DELETE:  # DELETE ANNOTATION
                        try:

                            session_state.annotation.result = session_state.annotation.delete_annotation(
                                conn).result

                            log_info(
                                f"Delete annotations for Task {session_state.task.name} with Annotation ID: {session_state.annotation.id}")
                            # st.json(session_state.annotation.result)
                        except Exception as e:
                            log_error(f"{e}: Delete annotation error")

                    else:
                        pass
                    # session_state.labelling_interface = None
                    # st.write(
                    #     f"After result: {session_state.annotation.result}")
                else:
                    if flag == EditorFlag.START:  # LOAD EDITOR
                        log_info("Editor Loaded")
                        pass

                    elif flag == EditorFlag.SKIP:  # NEW ANNOTATION
                        try:

                            skip_return = session_state.annotation.skip_task(
                                skipped=True, conn=conn)

                            log_info(
                                f"Skip for Task {session_state.task.name} with Annotation ID: {session_state.annotation.id}\n{skip_return}")
                        except Exception as e:
                            log_error(e)
                    else:
                        pass

        # annotations = session_state.annotation.query_annotations()
        # log_info("Query in main")
        annotations_dict = session_state.annotation.generate_annotation_dict()
        task = session_state.task.generate_editor_format(
            annotations_dict=annotations_dict, predictions_dict=None)
        # st.write(annotations_dict)
        # st.write(f"Before result: {session_state.annotation.result}")

    # Load empty if no data selected TODO: if remove Confirm button -> faster UI but when rerun immediately -> doesn't require loading of buffer editor
    except Exception as e:
        log_error(
            f"{e}: No data selected. Could not generate editor format based on task")
        # Preload blank editor
        task = {
            "annotations":
            [],
                'predictions': [],
                'id': 1,
                'data': {
                    # 'image': "https://app.heartex.ai/static/samples/sample.jpg"
                    'image': load_buffer_image()
            }
        }
    # st.json(task)

    if "labelling_interface" in session_state:
        del session_state.labelling_interface

    st.json(user)
    st.write(f"{session_state.project.deployment_type}")
    if session_state.editor.editor_config:
        with col2:
            try:

                log_info("Loading Image Labelling Interface")
                # result, flag = EDITOR_CONFIG.get(session_state.project.deployment_type, DetectionBBOX)(
                #     session_state.editor.editor_config, user, task)
                session_state.new_annotation_flag = 1  # UNLOCK for changes

# TODO: #12 Component doesn't load on second run -> only in incognito (no caching)
                result_raw = st_labelstudio(
                    session_state.editor.editor_config, interfaces, user, task, key="labelling_interface")

# ********************************************************************************************************
            except KeyError as e:
                log_error(f"Editor {e}")

            if "labelling_interface" in session_state:
                del session_state.labelling_interface
                # annotations_dict = session_state.annotation.generate_annotation_dict()
                # task = session_state.task.generate_editor_format(
                #     annotations_dict=annotations_dict, predictions_dict=None)

    # >>>> IF editor XML config fails to be loaded into Editor Class / not available in DB
    else:
        editor_no_load_warning = f"Image Labelling Interface failed to load"
        log_error(editor_no_load_warning)
        st.error(editor_no_load_warning)
# NOTE: Load ^ into results.py

# *************************EDITOR**********************************************

    col1, col2, col3 = st.beta_columns(3)
    # col1.write(vars(session_state.project))
    # # col1.write(session_state.project.dataset_list['My Third Dataset'])
    # col2.write(vars(session_state.editor))
    # # col3.write(vars(session_state.task))
    # st.write(vars(session_state.user))


def main():
    show()


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
