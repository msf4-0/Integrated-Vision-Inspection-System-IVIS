"""
Title: New Dataset Page
Date: 7/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)

Copyright (C) 2021 Selangor Human Resource Development Centre

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. 

Copyright (C) 2021 Selangor Human Resource Development Centre
SPDX-License-Identifier: Apache-2.0
========================================================================================

"""

import sys
from copy import deepcopy
from enum import IntEnum
from pathlib import Path
from time import perf_counter, sleep
from typing import Dict, Union
from humanize import naturalsize
import streamlit as st
from stqdm import stqdm
from streamlit import cli as stcli
from streamlit import session_state


# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

for path in sys.path:
    if str(LIB_PATH) not in sys.path:
        sys.path.insert(0, str(LIB_PATH))  # ./lib
    else:
        pass

from core.utils.code_generator import get_random_string
from core.utils.helper import check_filetype
from core.utils.log import logger
from core.webcam import webcam_webrtc
from data_manager.database_manager import init_connection
from data_manager.dataset_management import NewDataset, get_dataset_name_list, query_dataset_list
from path_desc import chdir_root
from project.project_management import NewProject, Project, ProjectPagination
from annotation.annotation_management import NewAnnotations, Task
from user.user_management import User

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration >>>>
# new_dataset = {}  # store
place = {}
DEPLOYMENT_TYPE = ("", "Image Classification", "Object Detection with Bounding Boxes",
                   "Semantic Segmentation with Polygons")


class DeploymentType(IntEnum):
    Image_Classification = 1
    OD = 2
    Instance = 3
    Semantic = 4

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return DeploymentType[s]
        except KeyError:
            raise ValueError()


def new_dataset(RELEASE=True, conn=None):
    if not conn:
        conn = init_connection(**st.secrets["postgres"])

    chdir_root()  # change to root directory

    # ******** DEBUGGING UPLOADER ONLY ********
    # NOTE: This will not work for the debugging of inserting uploaded annotations because
    #  it requires an existing Project instead of a NewProject
    if not RELEASE:
        if "new_project" not in session_state:
            session_state.new_project = NewProject(get_random_string(length=8))
            # session_state.new_project.deployment_type = "Object Detection with Bounding Boxes"
            # session_state.new_project.deployment_type = "Image Classification"
            session_state.new_project.deployment_type = "Semantic Segmentation with Polygons"
    # ******** DEBUGGING ********

    # ******** SESSION STATE ********

    if "new_dataset" not in session_state:
        # set random dataset ID before getting actual from Database
        logger.info("Enter new dataset")
        session_state.new_dataset = NewDataset(get_random_string(length=8))
        session_state.data_source_radio = "File Upload 📂"
    if 'user' not in session_state:
        session_state.user = User(1)
    # ******** SESSION STATE ********

    # >>>>>>>> New Dataset INFO >>>>>>>>
    # Page title
    if session_state.is_labeled:
        st.write(
            f"# __Deployment Type: {session_state.new_project.deployment_type}__")
        st.write("## __Upload Labeled Dataset__")
    else:
        st.write("# __Add New Dataset__")
    st.markdown("___")

    # right-align the dataset ID relative to the page
    _, id_right = st.columns([3, 1])
    id_right.write(
        f"### __Dataset ID:__ {session_state.new_dataset.dataset_id}")

    outercol1, outercol2, outercol3 = st.columns([1.5, 3.5, 0.5])

    # >>>>>>> DATASET INFORMATION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    outercol1.write("## __Dataset Information :__")

    # >>>> CHECK IF NAME EXISTS CALLBACK >>>>
    def check_if_name_exist(field_placeholder, conn):
        context = {'column_name': 'name',
                   'value': session_state.name}
        if session_state.name:
            if session_state.new_dataset.check_if_exists(context, conn):
                session_state.new_dataset.name = None
                field_placeholder['name'].error(
                    f"Dataset name used. Please enter a new name")
                sleep(1)
                logger.error(f"Dataset name used. Please enter a new name")
            else:
                session_state.new_dataset.name = session_state.name
                logger.error(f"Dataset name fresh and ready to rumble")

    outercol2.text_input(
        "Dataset Title", key="name", help="Enter the name of the dataset", on_change=check_if_name_exist, args=(place, conn,))
    place["name"] = outercol2.empty()

    # **** Dataset Description (Optional) ****
    description = outercol2.text_area(
        "Description (Optional)", key="desc", help="Enter the description of the dataset")
    if description:
        session_state.new_dataset.desc = description
    else:
        pass

    # <<<<<<<< New Dataset INFO <<<<<<<<

    # >>>>>>>> New Dataset Upload >>>>>>>>

    outercol1, outercol2, outercol3 = st.columns([1.5, 3.5, 0.5])

    outercol1.write("## __Dataset Upload:__")

    if not session_state.is_labeled:
        data_source_options = ["Webcam 📷", "File Upload 📂"]
        # col1, col2 = st.columns(2)

        data_source = outercol2.radio(
            "Data Source", options=data_source_options, key="data_source_radio")
        data_source = data_source_options.index(data_source)
    else:
        data_source = 1

    outercol1, outercol2, outercol3 = st.columns([1.5, 2, 2])
    dataset_size_string = f"- ### Number of datas: **{session_state.new_dataset.dataset_size}**"
    dataset_filesize_string = f"- ### Total size of data: **{naturalsize(value=session_state.new_dataset.calc_total_filesize(),format='%.2f')}**"
    outercol3.markdown(" ____ ")

    dataset_size_place = outercol3.empty()
    dataset_size_place.write(dataset_size_string)

    dataset_filesize_place = outercol3.empty()
    dataset_filesize_place.write(dataset_filesize_string)

    outercol3.markdown(" ____ ")
    # TODO: #15 Webcam integration
    # >>>> WEBCAM >>>>
    if data_source == 0:
        with outercol2:
            webcam_webrtc.app_loopback()

    # >>>> FILE UPLOAD >>>>
    # TODO #24 Add other filetypes based on filetype table
    # Done #24

    elif data_source == 1:

        # uploaded_files_multi = outercol2.file_uploader(
        #     label="Upload Image", type=['jpg', "png", "jpeg", "mp4", "mpeg", "wav", "mp3", "m4a", "txt", "csv", "tsv"], accept_multiple_files=True, key="upload_widget", on_change=check_filetype_category, args=(place,))
        # allowed_types = ['jpg', "png", "jpeg", "mp4",
        #                  "mpeg", "wav", "mp3", "m4a", "txt", "csv", "tsv"]
        allowed_types = ['jpg', "png", "jpeg", "txt", "csv", "xml", "json"]
        uploaded_files_multi = outercol2.file_uploader(
            label="Upload Image", type=allowed_types, accept_multiple_files=True, key="upload_widget")
        # ******** INFO for FILE FORMAT **************************************
        with outercol1.expander("File Format Infomation", expanded=True):
            # not using these formats for our application
            # 2. Video: .mp4, .mpeg
            # 3. Audio: .wav, .mp3, .m4a
            # 4. Text: .txt, .csv
            file_format_info = """
            #### Image Format:
            Image: .jpg, .png, .jpeg
            """
            st.info(file_format_info)
            if session_state.is_labeled:
                st.info(
                    "#### Compatible Annotation Format:  \n"
                    "- Object detection should have one XML file for each uploaded image.  \n"
                    "- Image classification should only have images and only one CSV file, "
                    "the first row of CSV file should be the filename with extension, while "
                    "the second row should be the class label name.  \n"
                    "- Image segmentation should only have images and only one COCO JSON file.  \n")

        place["upload"] = outercol2.empty()
        if len(uploaded_files_multi) > 1:
            # outercol2.write("uploaded_files_multi")  # TODO Remove
            # outercol2.write(uploaded_files_multi)  # TODO Remove
            if session_state.is_labeled:
                with st.spinner("Checking uploaded dataset and annotations"):
                    logger.info(
                        "Checking uploaded dataset and annotations ...")
                    start = perf_counter()
                    uploaded_files_multi = session_state.new_dataset.validate_labeled_data(
                        uploaded_files_multi,
                        session_state.new_project.deployment_type)
                    time_elapsed = perf_counter() - start
                    logger.info(f"Done. [{time_elapsed:.4f} seconds]")

            else:
                check_filetype(
                    uploaded_files_multi, session_state.new_dataset, place)

            session_state.new_dataset.dataset = deepcopy(uploaded_files_multi)

            session_state.new_dataset.dataset_size = len(
                uploaded_files_multi)  # length of uploaded files
        else:
            session_state.new_dataset.dataset_size = 0  # length of uploaded files
            session_state.new_dataset.dataset = []
        dataset_size_string = f"- ### Number of datas: **{session_state.new_dataset.dataset_size}**"
        dataset_filesize_string = f"- ### Total size of data: **{naturalsize(value=session_state.new_dataset.calc_total_filesize(),format='%.2f')}**"

        # outercol2.write(uploaded_files_multi[0]) # TODO: Remove
        dataset_size_place.write(dataset_size_string)
        dataset_filesize_place.write(dataset_filesize_string)

        if uploaded_files_multi and session_state.is_labeled:
            outercol3.success(
                "Annotations are verified to be compatible and in the correct format.")

    # Placeholder for WARNING messages of File Upload widget

    # with st.expander("Data Viewer", expanded=False):
    #     imgcol1, imgcol2, imgcol3 = st.columns(3)
    #     imgcol1.checkbox("img1", key="img1")
    #     for image in uploaded_files_multi:
    #         imgcol1.image(uploaded_files_multi[1])

    # TODO: KIV

    # col1, col2, col3 = st.columns([1, 1, 7])
    # webcam_button = col1.button(
    #     "Webcam 📷", key="webcam_button", on_click=update_webcam_flag)
    # file_upload_button = col2.button(
    #     "File Upload 📂", key="file_upload_button", on_click=update_file_uploader_flag)

    # <<<<<<<< New Dataset Upload <<<<<<<<
    # ******************************** SUBMISSION *************************************************
    success_place = st.empty()
    context = {'name': session_state.new_dataset.name,
               'upload': session_state.new_dataset.dataset}

    # st.write("context")
    # st.write(context)
    submit_col1, submit_col2 = st.columns([3, 0.5])
    submit_button = submit_col2.button("Submit", key="submit")

    if submit_button:
        keys = ["name", "upload"]
        session_state.new_dataset.has_submitted = session_state.new_dataset.check_if_field_empty(
            context, field_placeholder=place, name_key='name')

        if session_state.new_dataset.has_submitted:

            if session_state.new_dataset.save_dataset():

                success_place.success(
                    f"Successfully created **{session_state.new_dataset.name}** dataset")

                if session_state.new_dataset.insert_dataset():

                    success_place.success(
                        f"Successfully stored **{session_state.new_dataset.name}** dataset information in database")

                    if session_state.is_labeled:
                        if 'project' not in session_state:
                            # We need to insert the project_dataset here after the dataset
                            # has been stored
                            with st.spinner("Initializing the project with the uploaded labeled dataset ..."):
                                existing_dataset, _ = query_dataset_list()
                                dataset_dict = get_dataset_name_list(
                                    existing_dataset)
                                # add the uploaded dataset as the dataset_chosen, to
                                # allow the insert_project_dataset to work
                                session_state.new_project.dataset_chosen = [
                                    session_state.new_dataset.name]
                                session_state.new_project.insert_project_dataset(
                                    dataset_dict)

                            project_id = session_state.new_project.id
                            del session_state['new_project']
                            session_state.project = Project(project_id)
                            logger.info(f"Project ID {project_id} initialized")

                        with st.spinner("Querying all the labeled images ..."):
                            all_task, all_task_column_names = Task.query_all_task(
                                session_state.project.id,
                                return_dict=True,
                                # using True to use 'id' instead of 'ID' for the first column name
                                for_data_table=True)
                            task_df = Task.create_all_task_dataframe(
                                all_task, all_task_column_names)

                            # taking only the Path stem to consider the case of Label Studio exported
                            #  XML filenames without any file extension
                            if session_state.project.deployment_type == 'Object Detection with Bounding Boxes':
                                task_df.iloc[:, 0] = task_df.iloc[:, 0].apply(
                                    lambda x: Path(x).stem)

                        total_images = len(task_df)
                        filetype = session_state.new_dataset.filetype
                        logger.info("Submitting uploaded annotations ...")
                        start_t = perf_counter()
                        with st.spinner("Inserting uploaded annotations into database ..."):
                            result_generator = session_state.new_dataset.annotation_parser(
                                session_state.project.deployment_type
                            )
                            for img_name, result in stqdm(result_generator,
                                                          total=total_images,
                                                          unit=filetype):
                                start_task = perf_counter()
                                logger.debug(f"Image name {img_name}")
                                logger.debug(f"Annotation result {result}")

                                task_row = task_df.loc[
                                    task_df['Task Name'] == img_name
                                ].to_dict(orient='records')[0]

                                start = perf_counter()
                                task = Task(task_row,
                                            session_state.project.dataset_dict,
                                            session_state.project.id)
                                total = perf_counter() - start
                                logger.debug(
                                    f"Task instantiated [{total:.4f}s]")

                                start = perf_counter()
                                annotation = NewAnnotations(
                                    task, session_state.user)
                                total = perf_counter() - start
                                logger.debug(
                                    f"Annotation instantiated [{total:.4f}s]")

                                # Submit annotations to DB
                                start = perf_counter()
                                annotation.submit_annotations(
                                    result, session_state.user.id, conn)
                                total = perf_counter() - start
                                logger.debug(
                                    f"Annotation ID {annotation.id} submitted [{total:.4f}s]")

                                time_elapsed = perf_counter() - start_task
                                logger.info(
                                    f"Annotation submitted successfully [{time_elapsed:.4f}s]")

                        time_elapsed = perf_counter() - start_t
                        st.success(
                            "🎉 Successfully stored all the uploaded annotations!")
                        logger.info("Successfully inserted all annotations for "
                                    f"{len(task_df)} images into database. "
                                    f"Took {time_elapsed:.2f} seconds. "
                                    f"Average {time_elapsed / len(task_df):.4f}s per image")

                        with st.spinner("Updating project labels and editor configuration ..."):
                            default_labels = list(
                                session_state.project.editor.get_labels())
                            logger.debug("Default labels found in "
                                         f"editor config: {default_labels}")
                            project_id = session_state.project.id
                            # get the unique new labels from all the annotations
                            new_labels = session_state.project.get_existing_unique_labels(
                                project_id)

                            # update editor_config with the new labels from the uploaded annotations
                            for label in new_labels:
                                newChild = session_state.project.editor.create_label(
                                    'value', label)
                                logger.debug(
                                    f"newChild: {newChild.attributes.items()}")
                                session_state.project.editor.labels = session_state.project.editor.get_labels()
                            logger.info(
                                f"New labels added: {session_state.project.editor.labels}")

                            # remove the default_labels came with the original editor_config
                            for label in default_labels:
                                logger.debug(f"Removing label: {label}")
                                session_state.project.editor.labels.remove(
                                    label)
                                removedChild = session_state.project.editor.remove_label(
                                    'value', label)
                                logger.debug(f"removedChild: {removedChild}")
                            session_state.project.editor.labels.sort()
                            logger.debug(f"After removing default labels: "
                                         f"{session_state.project.editor.labels}")

                            session_state.project.editor.update_editor_config()

                        # go directly to existing project dashboard
                        NewProject.reset_new_project_page()
                        NewDataset.reset_new_dataset_page()
                        session_state.project_pagination = ProjectPagination.Existing
                        st.experimental_rerun()

                else:
                    st.error(
                        f"Failed to stored **{session_state.new_dataset.name}** dataset information in database")
            else:
                st.error(
                    f"Failed to created **{session_state.new_dataset.name}** dataset")

    # st.write("vars(session_state.new_dataset)")
    # st.write(vars(session_state.new_dataset))


def main(RELEASE=False, conn=None):
    new_dataset(RELEASE, conn=conn)


if __name__ == "__main__":
    # DEFINE wide page layout for debugging on this page directly
    layout = 'wide'
    st.set_page_config(page_title="Integrated Vision Inspection System",
                       page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

    if st._is_running_with_streamlit:
        # debugging upload dataset
        session_state.is_labeled = True

        # initialise connection to Database
        conn = init_connection(**st.secrets["postgres"])

        main(RELEASE=False, conn=conn)
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
