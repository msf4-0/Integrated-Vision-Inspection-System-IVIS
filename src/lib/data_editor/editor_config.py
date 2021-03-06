"""
Title: Editor Config
Date: 22/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
import sys
from pathlib import Path
from typing import Union, List, Dict
from time import sleep
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state

# DEFINE Web APP page configuration
# layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
# LIB_PATH = SRC / "lib"
# if str(LIB_PATH) not in sys.path:
#     sys.path.insert(0, str(LIB_PATH))  # ./lib

# >>>> User-defined Modules >>>>
from data_editor.label_studio_editor_component.label_studio_editor import labelstudio_editor
from path_desc import chdir_root
from core.utils.log import logger  # logger
from data_manager.database_manager import init_connection
from data_editor.editor_management import load_sample_image
from annotation.annotation_management import LabellingPagination
from project.project_management import ExistingProjectPagination, NewProject, Project, ProjectPagination, ProjectPermission
# <<<< User-defined Modules <<<<

# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])
place = {}


def editor_config(project: Union[NewProject, Project]):
    logger.debug("Entered editor config")

    # project_id: int, deployment_type: str
    chdir_root()  # change to root directory

    # >>>> LOAD SAMPLE IMAGE >>>>
    data_url = load_sample_image()

# *********************** EDITOR SETUP ****************************************************
    interfaces = [
        "panel",
        "controls",
        "side-column"

    ]

    user = {
        'pk': 1,
        'firstName': "John",
        'lastName': "Snow"
    }

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
    # deployment_type = DEPLOYMENT_TYPE[project.deployment_type]
    # if "editor" not in session_state:

    #     project.editor = Editor(project.id, project.deployment_type)

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
    # lowercol1, lowercol2 = st.columns([1, 2])

    with col1:

        # >>>> LOAD LABELS
        project.editor.labels = project.editor.get_labels()
        # You only need to set the `multiselect` widget's `session_state` key to be equal
        # to the label options to allow the `session_state` to be updated at every refresh
        # NOTE: This is not needed anymore in version 0.89.0
        # session_state.labels_select = project.editor.labels

        # TODO remove
        tagName_attributes = project.editor.get_tagname_attributes(
            project.editor.childNodes)

        def add_label(place):

            if session_state.add_label and session_state.add_label not in project.editor.labels:
                logger.debug(f"{session_state.add_label = }")
                newChild = project.editor.create_label(
                    'value', session_state.add_label)
                logger.debug(f"newChild: {newChild.attributes.items()}")

                project.editor.labels = project.editor.get_labels()
                logger.info(f"New label added: {project.editor.labels}")

            elif session_state.add_label in project.editor.labels:
                label_exist_msg = f"Label '{session_state.add_label}' already exists in {project.editor.labels}"
                logger.error(label_exist_msg)
                place["add_label"].error(label_exist_msg)
                sleep(1)
                place["add_label"].empty()

        def update_labels(place):

            diff_12 = set(project.editor.labels).difference(
                session_state.labels_select)  # set 1 - set 2 REMOVAL
            diff_21 = set(session_state.labels_select).difference(
                project.editor.labels)  # set 2 - set 1 ADDITION
            if diff_12:
                logger.debug("Removal")
                removed_label = list(diff_12).pop()

                # to avoid removing existing labels used for annotating!
                if isinstance(project, Project):
                    existing_annotations = project.get_existing_unique_labels()
                    if removed_label in existing_annotations:
                        place["warning_label_removal"].error(
                            f"WARNING: You are trying to remove a label '{removed_label}' "
                            "that has been used to label your images before!")
                        sleep(3)
                        place["warning_label_removal"].empty()
                        return

                try:
                    logger.debug(f"Removing: {removed_label}")
                    project.editor.labels.remove(
                        removed_label)
                    project.editor.labels.sort()
                    # TODO: function to remove DOM
                    removedChild = project.editor.remove_label(
                        'value', removed_label)
                    logger.debug(f"removedChild: {removedChild}")
                    logger.info(f"Label removed {project.editor.labels}")

                except ValueError as e:
                    st.error(f"{e}: Label already removed")
            elif diff_21:
                print("Addition")
                # added_label = list(diff_21).pop()
                # project.editor.labels.append(added_label)
                project.editor.labels = sorted(
                    project.editor.labels + list(diff_21))
                # TODO: function to add DOM

            else:
                print("No Change")
            project.editor.labels = project.editor.get_labels()

            # if len(project.editor.labels) > len(session_state.labels_select):
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
        logger.debug(f"Before multi {project.editor.labels}")
        # >>>>>>> REMOVE LABEL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        st.multiselect('Labels', options=project.editor.labels, default=project.editor.labels,
                       key='labels_select', on_change=update_labels, args=(place,))
        # to show the error message when trying to remove existing labels
        place["warning_label_removal"] = st.empty()

        def save_editor_config():
            if not session_state.labels_select:
                place["warning_label_removal"].error(
                    "Please provide at least one label")
                st.stop()

            logger.info("Updating Editor Config......")

            if project.editor.update_editor_config():
                st.success(f"Successfully updated editor configuration")

        if st.button('Save', key='save_editor_config'):
            save_editor_config()

            if session_state.get('new_project_pagination'):
                # user just finished creating a new project, so enter the project directly
                new_project_id = session_state.new_project.id
                NewProject.reset_new_project_page()

                session_state.project_pagination = ProjectPagination.Existing
                session_state.project_status = ProjectPagination.Existing
                session_state.append_project_flag = ProjectPermission.ViewOnly

                # directly go to labelling page and start labelling with Label Studio Editor
                session_state.existing_project_pagination = ExistingProjectPagination.Labelling
                session_state.labelling_pagination = LabellingPagination.Editor
                # this session_state is to enable the feature for auto next image
                session_state.show_next_unlabeled = True

                session_state.project = Project(new_project_id)
                logger.info(f"Entering Project {new_project_id}")
                st.experimental_rerun()

        # >>>>>>>>>> TODO #66 Add Color picker for Bbox, Segmentation Polygons and Segmentation Masks >>>>>>>>>>>>>>
    # with lowercol1:
    #     st.write("Labels selected:")
    #     st.write(session_state.labels_select)
    #     st.write("Doc")
    #     st.write(project.editor.xml_doc)
    #     st.write("Label Childnodes")
    #     st.write(project.editor.childNodes)
    #     st.write("Labels")
    #     st.write(project.editor.labels)
    #     st.write("Attributes")
    #     st.write(tagName_attributes)
    #     st.write("Editor Class")
    #     st.write(vars(project.editor))

    with st.expander('Editor Config', expanded=False):
        config2 = project.editor.to_xml_string(
            pretty=True)
        st.code(config2, language='xml')

    with col2:
        # st.text_input("Check column", key="column2")
        labelstudio_editor(config2, interfaces, user,
                           task, key='editor_config')


def main(RELEASE=True):
    # ****************** TEST ******************************
    if not RELEASE:
        logger.debug("At main")
        # ************************TO REMOVE************************
        with st.sidebar.container():
            st.image("resources/MSF-logo.gif", use_column_width=True)
            st.title("Integrated Vision Inspection System", anchor='title')
            st.header(
                "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
            st.markdown("""___""")
        # ************************TO REMOVE************************
        # project_id = 7
        # # get enum ->2
        # deployment_type = DEPLOYMENT_TYPE["Object Detection with Bounding Boxes"]
        if 'project' not in session_state:
            session_state.project = Project(103)
        # project = Project(7)
        # project.refresh_project_details()
        editor_config(session_state.project)
        # st.write(vars(session_state.project))
        st.write(vars(session_state.project.editor))


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        # False for debugging on this page
        main(RELEASE=False)
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
