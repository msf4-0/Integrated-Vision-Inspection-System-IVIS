"""
Title: Data Labelling
Date: 15/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
import copy
import sys
from pathlib import Path
from typing import List
from threading import Thread
from time import perf_counter, sleep
from copy import deepcopy
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state
# DEFINE Web APP page configuration
layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from core.utils.log import logger  # logger
from project.project_management import Project
from data_editor.editor_management import EditorFlag
from user.user_management import User
from data_manager.database_manager import init_connection
from annotation.annotation_management import (LabellingPagination, Annotations, NewAnnotations, Task,
                                              load_buffer_image, task_labelling_columns,
                                              get_task_row, reset_editor_page)
from data_editor.label_studio_editor_component.label_studio_editor import labelstudio_editor
from data_manager.data_table_component.data_table import data_table

# <<<< User-defined Modules <<<<
conn = init_connection(**st.secrets["postgres"])

# NOTE: not used********************************************
from streamlit.report_thread import add_report_ctx


def editor(data_id: List = []):
    """Page for labeling with Label Studio Editor

    `data_id`: the selected task ID from the data_table.
    """
    logger.debug("Inside Editor function")
    chdir_root()  # change to root directory

    # ******** SESSION STATE ***********************************************************

    if "labelling_interface" not in session_state:
        session_state.labelling_interface = ([], 0)
    if "new_annotation_flag" not in session_state:
        session_state.new_annotation_flag = 0
    if "data_labelling_table" not in session_state:
        session_state.data_labelling_table = data_id
    if "labelling_prev_result" not in session_state:
        session_state.labelling_prev_result = []
    if "show_next_unlabeled" not in session_state:
        # a flag to decide whether to show next unlabeled data
        # if this is True, `st.dataframe` will be used instead of clickable `data_table`
        session_state.show_next_unlabeled = False

    # ******** SESSION STATE *********************************************************

# ************************************** COLUMN PLACEHOLDERS***************************************
    back_to_labelling_dashboard_button_place = st.empty()
    note_place = st.empty()
    title_place = st.empty()
    title_place.write("### **Data Labelling**")
    if session_state.show_next_unlabeled:
        main_col2 = st.empty()
    else:
        main_col1, main_col2 = st.columns([2.5, 3])
    # main_col1.write("### **Data Labelling**")
# ************************************** COLUMN PLACEHOLDERS***************************************

# ************************** BACK TO LABELLING DASHBOARD CALLBACK******************************
    def to_labelling_dashboard_page():
        session_state.labelling_pagination = LabellingPagination.AllTask
        reset_editor_page()
        logger.debug(
            f"Returning to labelling dashboard: {session_state.labelling_pagination}")

    back_to_labelling_dashboard_button_place.button("Return to Labelling Dashboard",
                                                    key='back_to_labelling_dashboard_page',
                                                    on_click=to_labelling_dashboard_page)


# ************************** BACK TO LABELLING DASHBOARD CALLBACK******************************

# ************************** DATA TABLE ***********************************************

    all_task, all_task_column_names = Task.query_all_task(session_state.project.id,
                                                          return_dict=True, for_data_table=True)
    task_df = Task.create_all_task_dataframe(
        all_task, all_task_column_names)

    def load_data(task_df):
        logger.debug(f"Inside load data CALLBACK")
        if session_state.data_labelling_table:
            task_id = session_state.data_labelling_table[0]
            logger.debug(f"{task_id = }")
            task_row = get_task_row(task_id, task_df)

        if "labelling_interface" in session_state:
            logger.debug("Deleting session_state.labelling_interface")
            del session_state.labelling_interface

        # >>>> INSTANTIATE TASK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            if 'task' not in session_state:
                session_state.task = Task(
                    task_row, session_state.project.dataset_dict, session_state.project.id)
            else:
                session_state.task = Task(
                    task_row, session_state.project.dataset_dict, session_state.project.id)

            # NOTE ************************TEST**************************************
            session_state.labelling_prev_result = []
            logger.info(
                f"Task instantiated for id: {session_state.task.id} for {session_state.task.name}🏃")
        # >>>> INSTANTIATE TASK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            # Check if annotation exist
            if Annotations.check_if_annotation_exists(session_state.task.id, session_state.project.id, conn):

                # Annotation exist if task is labelled
                # if session_state.task.is_labelled:
                session_state.annotation = Annotations(
                    session_state.task)
                logger.info(
                    f"Annotation {session_state.annotation.id} exists for Task ID: {session_state.task.id} for {session_state.task.name}")
            else:
                session_state.annotation = NewAnnotations(
                    session_state.task, session_state.user)
        else:
            logger.error(f"task_id NaN->Task not loaded😫")

        # set FLAG = 1 such that init render of none will be ignored
        # TODO Reset to 0 / del when switching back to dashboard
        session_state.new_annotation_flag = 1

    # st.write(session_state.new_annotation_flag)

# ************************ FIRST RENDER: ********************************************************
    if session_state.new_annotation_flag == 0:

        if session_state.data_labelling_table:  # if task id passed as argument
            load_data(task_df)

        elif session_state.data_labelling_table == []:  # if task id NOT passed as argument
            session_state.data_labelling_table = [
                all_task[0]['id']]  # set value as first ID
            load_data(task_df)
# ************************ FIRST RENDER: ********************************************************


# ************************ NOTES about the feature of auto-next task ****************************
    with note_place.expander("NOTES about the feature of auto move to next unlabeled task"):
        st.info("""If you want to make use of the this feature, click the "Start Labelling"
        button and start labelling, then the image will automatically move to the next
        one after you submitted. If you experience any problem, click the
        "Return to Labelling Dashboard" then click the "Start Labelling" button again.
        """)


# TODO Fix Data Table is_labelled not updated at re-run
# ************************** DATA TABLE ********************************************************
    if not session_state.show_next_unlabeled:
        # only show the data_table when not using the auto-next-task feature
        with main_col1:
            data_table(all_task, task_labelling_columns,
                       checkbox=False, key='data_labelling_table',
                       on_change=load_data, args=(task_df,))

        # >>>>> Temp Image Viewer >>>>>
        # st.write(all_task[0])
        # st.write(session_state.data_labelling_table)
        # st.write(vars(session_state.project))


# ************************** DATA TABLE ********************************************************

# *************************EDITOR**********************************************

# ***************INTERFACE CONFIG**********************************************
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
    ]
# ***************INTERFACE CONFIG**********************************************

    # >>>> User Informations >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    user = {
        'pk': session_state.user.id,
        'firstName': session_state.user.first_name,
        'lastName': session_state.user.last_name
    }

    try:
        # if session_state.labelling_interface[0] and session_state.new_annotation_flag != 0:
        if session_state.new_annotation_flag == 1:

            # ********************************** CRUD CALLBACK for annotation results *********************************************

            if "labelling_interface" in session_state:
                # This IF STATEMENT included as labelling interface reset at every change in data selection
                # st.write("Inside extra function")

                if (session_state.labelling_interface != session_state.labelling_prev_result):
                    # Compare present Editor component state with prev results state

                    # >>>> ASSIGN RESULTS TO INTERFACE >>>>>
                    results = session_state.labelling_interface
                    # st.write("Inside extra result update function")
                    result, flag = results if results else (None, None)
                    logger.debug(f"Flag at main: {flag}")
                    # st.write("Result", result)

                    # >>>> IF results exists => if there is submission / update
                    if result:
                        # st.write(
                        #     "Inside extra result update function results exist")

                        if flag == EditorFlag.START:  # LOAD EDITOR
                            logger.debug("Editor Loaded (In result)")
                            pass

                        elif flag == EditorFlag.SUBMIT:  # NEW ANNOTATION
                            try:

                                # Submit annotations to DB
                                session_state.annotation.result = session_state.annotation.submit_annotations(
                                    result, session_state.user.id, conn)

                                # Memoir prev results
                                session_state.labelling_prev_result = session_state.labelling_interface
                                logger.debug(
                                    f'{session_state.annotation.result}')
                                logger.info(
                                    f"New submission for Task {session_state.task.name} with Annotation ID: {session_state.annotation.id}")

                                # session_state.show_next_unlabeled = True

                            except Exception as e:
                                logger.error(f"{e}: New Annotation error")

                        elif flag == EditorFlag.UPDATE:  # UPDATE ANNOTATION
                            try:

                                session_state.labelling_prev_result = session_state.labelling_interface

                                session_state.annotation.result = session_state.annotation.update_annotations(
                                    result, session_state.user.id, conn).result

                                logger.info(
                                    f"Update annotations for Task {session_state.task.name} with Annotation ID: {session_state.annotation.id}")

                            except Exception as e:
                                logger.error(f"{e}: Update annotation error")

                        elif flag == EditorFlag.DELETE:  # DELETE ANNOTATION
                            try:

                                session_state.labelling_prev_result = session_state.labelling_interface

                                session_state.annotation.result = session_state.annotation.delete_annotation(
                                    conn).result

                                logger.info(
                                    f"Delete annotations for Task {session_state.task.name} with Annotation ID: {session_state.annotation.id}")

                            except Exception as e:
                                logger.error(f"{e}: Delete annotation error")

                        else:
                            pass

                        st.experimental_rerun()

                    else:
                        if flag == EditorFlag.START:  # LOAD EDITOR
                            logger.debug("Editor Loaded")
                            pass

                        elif flag == EditorFlag.SKIP:  # SKIP ANNOTATION
                            try:

                                skip_return = session_state.annotation.skip_task(
                                    skipped=True, conn=conn)

                                logger.info(
                                    f"Skip for Task {session_state.task.name} with Annotation ID: {session_state.annotation.id}\n{skip_return}")

                                # session_state.show_next_unlabeled = True
                                st.experimental_rerun()

                            except Exception as e:
                                logger.error(e)
                        else:
                            pass

                    # with main_col1:
                    #     st.write(results)
                    #     st.write(flag)
            # ********************************** CRUD CALLBACK for annotation results *********************************************
            annotations_dict = session_state.annotation.generate_annotation_dict()
            task = session_state.task.generate_editor_format(
                annotations_dict=annotations_dict, predictions_dict=None)

       # *************************************** LABELLING INTERFACE *******************************************
            with main_col2.container():
                logger.debug("Showing LABEL STUDIO EDITOR INTERFACE")
                st.write(
                    f"### **{session_state.task.filetype.name}: {session_state.task.name}**")
                labelstudio_editor(
                    session_state.project.editor.editor_config, interfaces, user, task, key="labelling_interface")
                # temporary workaround to make sure labelstudio_editor does not get covered
                st.markdown("""
                <style>
                iframe {
                    height: calc(200vh + 500px);
                }
                </style>
                """, unsafe_allow_html=True)
        # *************************************** LABELLING INTERFACE *******************************************

        # Load empty if no data selected TODO: if remove Confirm button -> faster UI but when rerun immediately -> doesn't require loading of buffer editor
    except Exception as e:
        logger.error(
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

# ************************ AUTO NEXT TASK ********************************************************
# NOTE: This method makes the Label Studio Editor's component sometimes get hidden
#  within the small `div` container after a submission,
#  so it does not work properly for now....
# SOLUTION: Use st.dataframe at the side instead of rendering only the Label Studio Editor
#  - OR use the st.markdown to change the CSS styling for `iframe` tag, which is the tag
#    of the Label Studio Editor HTML component

# only show the Label Studio interface for labeling, without data_table
    if session_state.show_next_unlabeled:
        # automatically move the labeling interface to the next unlabeled task
        st.markdown("___")
        if session_state.data_labelling_table:
            current_id = session_state.data_labelling_table[0]
        else:
            current_id = task_df.iloc[0, 0]
        unlabeled_task_ids = task_df.query(
            "`Is Labelled` == False and Skipped == False")['id']
        # only do this if there is still unlabeled task
        if not unlabeled_task_ids.empty:
            # must use `int` to change it from `numpy.int` dtypes
            next_unlabeled_task_id = int(unlabeled_task_ids.values[0])
            # set this to show the next task, must set it to a list as this is how the `data_table` returns
            session_state.data_labelling_table = [next_unlabeled_task_id]
            if next_unlabeled_task_id != current_id:
                logger.info(
                    f"Proceeding to next task ID: {next_unlabeled_task_id}")
                load_data(task_df)
                logger.debug("REFRESHINGGG")
                st.experimental_rerun()
        else:
            logger.info("All tasks labeled successfully for Project ID: "
                        f"{session_state.project.id}")
            note_place.success("🎉 **You have labeled all tasks!**")
            st.balloons()
            # clear the title and Label Studio Editor
            title_place.empty()
            main_col2.empty()
            st.stop()
# ************************ AUTO NEXT TASK ********************************************************


# ********************************************************************************************************


def index():
    RELEASE = True

    # ****************** TEST ******************************
    if not RELEASE:
        logger.debug("At Labelling INDEX")

        # ************************TO REMOVE************************
        with st.sidebar.container():
            st.image("resources/MSF-logo.gif", use_column_width=True)
            st.title("Integrated Vision Inspection System", anchor='title')
            st.header(
                "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
            st.markdown("""___""")

        # ************************TO REMOVE************************
        project_id_tmp = 43
        logger.debug(f"Entering Project {project_id_tmp}")

        # session_state.append_project_flag = ProjectPermission.ViewOnly

        if "project" not in session_state:
            # Editor will be instantiated inside Project class at same instance
            session_state.project = Project(project_id_tmp)

            logger.debug(
                f"NOT RELEASE: Instantiating Project {session_state.project.name}")
        if 'user' not in session_state:
            session_state.user = User(1)

        # ****************************** HEADER **********************************************
        st.write(f"# {session_state.project.name}")

        project_description = session_state.project.desc if session_state.project.desc is not None else " "
        st.write(f"{project_description}")

        st.markdown("""___""")
        # ****************************** HEADER **********************************************

        st.write(f"## **Labelling Section:**")
        editor()


def main():
    index()


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
