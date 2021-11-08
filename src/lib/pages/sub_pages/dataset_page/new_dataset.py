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

import os
import sys
import shutil
from copy import deepcopy
from enum import IntEnum
from pathlib import Path
import tarfile
from time import perf_counter, sleep
from typing import Dict, Union
from zipfile import ZipFile
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
from core.utils.helper import Timer, check_filetype
from core.utils.file_handler import extract_archive, list_files_in_archived, check_image_files
from core.utils.log import logger
from core.webcam import webcam_webrtc
from data_manager.database_manager import init_connection
from data_manager.dataset_management import NewDataset, get_dataset_name_list, query_dataset_list
from path_desc import TEMP_DIR, chdir_root
from project.project_management import NewProject, NewProjectPagination, Project, ProjectPagination
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


def new_dataset(RELEASE=True, conn=None, is_updating=False):
    """Function for the page of creating new dataset or adding more images to an existing
    project dataset.

    `is_updating`: A flag of updating an existing dataset, i.e. to add more data to 
    an existing project dataset (which was selected in existing_project_dashboard.py).

    `session_state.is_labeled`: A flag to tell whether the user chooses to upload a
    labeled dataset, then validation will also be required by using
    `NewDataset.validate_labeled_data()`
    """
    # NOTE:

    if not conn:
        conn = init_connection(**st.secrets["postgres"])

    chdir_root()  # change to root directory

    # ******** SESSION STATE ********
    if "new_dataset" not in session_state:
        # set random dataset ID before getting actual from Database
        logger.debug("Enter new dataset")
        session_state.new_dataset = NewDataset(get_random_string(length=8))
    if 'user' not in session_state:
        session_state.user = User(1)
    if 'is_labeled' not in session_state:
        session_state.is_labeled = False
    if session_state.is_labeled and ('project' not in session_state):
        # initialize project with the inserted project, note that this only works
        #  after the new_project has been stored in database
        project_id = session_state.new_project.id
        del session_state['new_project']
        session_state.project = Project(project_id)
        logger.info(f"Project ID {project_id} initialized")
    # ******** SESSION STATE ********

    # ******** DEBUGGING ********
    # NOTE: If debugging for inserting uploaded annotations, you need to select
    #  an existing project_id
    if not RELEASE:
        if not session_state.is_labeled and ("new_project" not in session_state):
            session_state.new_project = NewProject(get_random_string(length=8))
            # session_state.new_project.deployment_type = "Object Detection with Bounding Boxes"
            # session_state.new_project.deployment_type = "Image Classification"
            session_state.new_project.deployment_type = "Semantic Segmentation with Polygons"
        if session_state.is_labeled and ('project' not in session_state):
            project_id = 4
            logger.debug(f"""Entering Project ID {project_id} for debugging
            uploading labeled dataset""")
            session_state.project = Project(project_id)
    # ******** DEBUGGING ********

    if is_updating:
        # session_state.dataset_chosen should be obtained from existing_project_dashboard
        dataset_info = session_state.project.dataset_dict[session_state.dataset_chosen]
        # set the info to be equal to new_dataset to make things easier
        session_state.new_dataset.id = dataset_info.ID
        session_state.new_dataset.name = dataset_info.Name
        session_state.new_dataset.desc = dataset_info.Description

    if 'project' in session_state:
        deployment_type = session_state.project.deployment_type
    elif 'new_project' in session_state:
        deployment_type = session_state.new_project.deployment_type

    # >>>>>>>> New Dataset INFO >>>>>>>>
    # Page title
    if session_state.is_labeled or is_updating:
        st.write(f"# __Deployment Type: {deployment_type}__")
    if not is_updating:
        if session_state.is_labeled:
            st.write("## __Upload Labeled Dataset__")
            logger.info("Upload labeled dataset")
        else:
            st.write("# __Add New Dataset__")
    st.markdown("___")

    # right-align the dataset ID relative to the page
    _, id_right = st.columns([3, 1])
    id_right.write(
        f"### __Dataset ID:__ {session_state.new_dataset.id}")

    outercol1, outercol2, outercol3 = st.columns([1.5, 3.5, 0.5])

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
                logger.info(f"Dataset name fresh and ready to rumble")

    # >>>>>>> DATASET INFORMATION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if not is_updating:
        outercol1.write("## __Dataset Information :__")
        outercol2.text_input(
            "Dataset Title", key="name", help="Enter the name of the dataset", on_change=check_if_name_exist, args=(place, conn,))
        place["name"] = outercol2.empty()

        # **** Dataset Description (Optional) ****
        description = outercol2.text_area(
            "Description (Optional)", key="desc", help="Enter the description of the dataset")
        if description:
            session_state.new_dataset.desc = description
    else:
        st.write("## __Current Project Dataset Information :__")
        st.markdown(f"**Dataset name:** {session_state.new_dataset.name}")
        st.markdown(
            f"**Dataset description:** {session_state.new_dataset.desc}")

        st.markdown("## __Adding data to existing project dataset__")
        labeled = st.radio(
            "Select type of dataset to upload", options=('Not Labeled', 'Labeled'),
            index=0, key='labeled'
        )
        session_state.is_labeled = True if labeled == 'Labeled' else False

    # <<<<<<<< New Dataset INFO <<<<<<<<

    # >>>>>>>> New Dataset Upload >>>>>>>>

    outercol1, outercol2, outercol3 = st.columns([1.5, 3.5, 0.5])

    outercol1.write("## __Dataset Upload:__")

    if not session_state.is_labeled:
        data_source_options = ["Webcam 📷", "File Upload 📂"]
        # col1, col2 = st.columns(2)

        data_source = outercol2.radio(
            "Data Source", options=data_source_options, key="data_source_radio",
            index=1)
        data_source = data_source_options.index(data_source)
    else:
        data_source = 1

    outercol1, outercol2, outercol3 = st.columns([1.5, 2, 2])

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
        # allowed_types = ['jpg', "png", "jpeg", "txt", "csv", "xml", "json"]
        allowed_types = ['zip', 'tar.gz', 'tar.xz', 'tar.bz2']
        uploaded_archive = outercol2.file_uploader(
            label="Upload Image", type=allowed_types, accept_multiple_files=False, key="upload_widget")
        # outercol2.info("""NOTE: When you are uploading a lot of files at once, the
        # Streamlit app may look like it's stuck but please wait for it to finish uploading.""")
        # ******** INFO for FILE FORMAT **************************************
        with outercol1.expander("File Format Infomation", expanded=True):
            # not using these formats for our application
            # 2. Video: .mp4, .mpeg
            # 3. Audio: .wav, .mp3, .m4a
            # 4. Text: .txt, .csv
            file_format_info = """
            #### Image Format:
            Image: .jpg, .png, .jpeg

            **NOTE**: Only a single zipfile or tarfile with all the required files inside can be accepted.
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
        if uploaded_archive:
            # st.write(len(uploaded_archive))
            # st.write(uploaded_archive)

            # NOTE: once you closed the Streamlit UploadedFile (archive file in this case),
            # note that the contents **will be empty** the next time you open it again,
            # unless you created a deepcopy beforehand. Or you may choose to manually
            # open and close the archive file without context manager. Alternatively,
            # you can run `uploaded_archive.seek(0)` every time you want to reopen
            # the archive file
            with outercol2:
                with st.spinner("Getting archive content names and size ..."):
                    filepaths, content_size = list_files_in_archived(
                        archived_filepath=uploaded_archive.name,
                        file_object=uploaded_archive,
                        return_content_size=True,
                        skip_dir=True)
                # check_filetype(
                #     uploaded_files_multi, session_state.new_dataset, place)

            # session_state.new_dataset.dataset = image_names

            # length of uploaded files
            num_files = len(filepaths)
        else:
            content_size = 0
            num_files = 0
            # session_state.new_dataset.dataset = []

        with outercol3:
            dataset_size_string = f"- ### Number of datas: **{num_files}**"
            # dataset_filesize_string = f"- ### Total size of data: **{naturalsize(value=session_state.new_dataset.calc_total_filesize(),format='%.2f')}**"
            dataset_filesize_string = ("- ### Total size of data: "
                                       f"**{naturalsize(value=content_size, format='%.2f')}**")
            st.markdown(" ____ ")
            st.write(dataset_size_string)
            st.write(dataset_filesize_string)
            st.markdown(" ____ ")

    # Placeholder for WARNING messages of File Upload widget

    # with st.expander("Data Viewer", expanded=False):
    #     imgcol1, imgcol2, imgcol3 = st.columns(3)
    #     imgcol1.checkbox("img1", key="img1")
    #     for image in uploaded_archive:
    #         imgcol1.image(uploaded_archive[1])

    # TODO: KIV

    # col1, col2, col3 = st.columns([1, 1, 7])
    # webcam_button = col1.button(
    #     "Webcam 📷", key="webcam_button", on_click=update_webcam_flag)
    # file_upload_button = col2.button(
    #     "File Upload 📂", key="file_upload_button", on_click=update_file_uploader_flag)

    # <<<<<<<< New Dataset Upload <<<<<<<<
    # ******************************** SUBMISSION *************************************************
    success_place = st.empty()

    # st.write("context")
    # st.write(context)
    submit_col1, submit_col2 = st.columns([3, 0.5])
    with submit_col2:
        # using a placeholder for button to be able to clear it out later
        button_place = st.empty()
        submit_button = button_place.button("Submit", key="submit")

    # st.write(session_state)

    if submit_button:
        context = {'name': session_state.new_dataset.name,
                   'upload': session_state.upload_widget}
        if is_updating:
            # don't need to check for dataset name for existing dataset
            del context['name']

        session_state.new_dataset.has_submitted = session_state.new_dataset.check_if_field_empty(
            context, field_placeholder=place, name_key='name')

        if session_state.new_dataset.has_submitted:
            if is_updating:
                logger.info("Checking for duplicated filenames")
                with outercol2:
                    with st.spinner("Checking for duplicated filenames ..."):
                        uploaded_files = [
                            os.path.basename(f) for f in filepaths]
                        existing_images = session_state.project.data_name_list[
                            session_state.dataset_chosen]
                        duplicates = set(uploaded_files).intersection(
                            existing_images)
                    if duplicates:
                        logger.error("Duplicated image filenames found")
                        st.error("Found image filenames in the archive that are identical to "
                                 "the current project dataset. Please make sure to use "
                                 "different names to avoid overwriting existing images.")
                        with st.expander("List of duplicates:"):
                            st.markdown("  \n".join(duplicates))
                        st.stop()

            with outercol2:
                if session_state.is_labeled:
                    with st.spinner("Checking uploaded dataset and annotations ..."):
                        logger.info(
                            "Checking uploaded dataset and annotations")
                        start = perf_counter()
                        image_paths = session_state.new_dataset.validate_labeled_data(
                            uploaded_archive,
                            filepaths,
                            deployment_type)
                        time_elapsed = perf_counter() - start
                        logger.info(
                            f"Done. [{time_elapsed:.4f} seconds]")
                        st.success("""🎉 Annotations are verified to be compatible and
                        in the correct format.""")
                else:
                    with st.spinner("Checking uploaded images ..."):
                        logger.info("Checking uploaded images")
                        image_paths = check_image_files(filepaths)

            session_state.new_dataset.dataset = image_paths

            # create a temporary directory for extracting the archive contents
            if TEMP_DIR.exists():
                shutil.rmtree(TEMP_DIR)
            os.makedirs(TEMP_DIR)
            session_state.new_dataset.archive_dir = TEMP_DIR

            with outercol3:
                with st.spinner("Extracting uploaded archive contents ..."):
                    extract_archive(TEMP_DIR, file_object=uploaded_archive)

            if session_state.new_dataset.save_dataset(session_state.new_dataset.archive_dir):

                success_place.success(
                    f"Successfully created **{session_state.new_dataset.name}** dataset")

                if is_updating:
                    dataset_func = session_state.new_dataset.update_dataset_size
                else:
                    dataset_func = session_state.new_dataset.insert_dataset

                if dataset_func():

                    success_place.success(
                        f"Successfully stored **{session_state.new_dataset.name}** dataset information in database")

                    if is_updating:
                        # insert the new image info into the `task` table for existing dataset
                        image_names = [os.path.basename(p)
                                       for p in image_paths]
                        session_state.project.insert_new_project_task(
                            session_state.new_dataset.name,
                            session_state.new_dataset.id,
                            image_names=image_names)

                    if session_state.is_labeled:
                        # We need to insert the project_dataset here after the dataset
                        # has been stored
                        if not is_updating:
                            # if not updating, then we insert the new project_dataset
                            with st.spinner("Inserting the project dataset ..."):
                                # similar to `new_project.py`
                                existing_dataset, _ = query_dataset_list()
                                dataset_dict = get_dataset_name_list(
                                    existing_dataset)
                                # add the uploaded dataset as the dataset_chosen, to
                                # allow the insert_project_dataset to work
                                dataset_name = session_state.new_dataset.name
                                dataset_chosen = [dataset_name]
                                session_state.project.insert_project_dataset(
                                    dataset_chosen, dataset_dict)
                                project_id = session_state.project.id
                                logger.info(f"Inserted project dataset '{dataset_name}' for "
                                            f"Project {project_id} into project_dataset table")
                                # must refresh all the dataset details
                                session_state.project.refresh_project_details()

                        with st.spinner("Querying all the labeled images ..."):
                            all_task, all_task_column_names = Task.query_all_task(
                                session_state.project.id,
                                return_dict=True,
                                # using True to use 'id' instead of 'ID' for the first column name
                                for_data_table=True)
                            task_df = Task.create_all_task_dataframe(
                                all_task, all_task_column_names)

                            # taking only the filename without extension to consider the
                            #  case of Label Studio exported XML filenames without
                            #  any file extension
                            if session_state.project.deployment_type == 'Object Detection with Bounding Boxes':
                                task_df['Task Name'] = task_df['Task Name'].apply(
                                    lambda filename: os.path.splitext(filename)[0])

                        total_images = len(task_df)
                        filetype = session_state.new_dataset.filetype
                        logger.info("Submitting uploaded annotations ...")

                        start_t = perf_counter()
                        error_imgs = []
                        # set this to False to check how long each process takes
                        disable_timer = True

                        result_generator = session_state.new_dataset.parse_annotation_files(
                            session_state.project.deployment_type
                        )
                        message = "Inserting uploaded annotations into database"
                        for img_name, result in stqdm(result_generator, total=total_images,
                                                      st_container=st.sidebar,
                                                      unit=filetype, desc=message):
                            start_task = perf_counter()

                            task_row = task_df.loc[
                                task_df['Task Name'] == img_name
                            ].to_dict(orient='records')

                            if task_row:
                                # index into the List of Dict of task
                                task_row = task_row[0]
                            else:
                                logger.error(f"""Image '{img_name}' not found. Probably
                                removed because unable to read the image. Skipping
                                from submitting to database.""")
                                # these images were skipped in dataset_PNG_encoding()
                                error_imgs.append(img_name)
                                continue

                            with Timer("Task instantiated", disable_timer):
                                task = Task(task_row,
                                            session_state.project.dataset_dict,
                                            session_state.project.id)

                            with Timer("Annotation instantiated", disable_timer):
                                annotation = NewAnnotations(
                                    task, session_state.user)

                            # Submit annotations to DB
                            with Timer(f"Annotation ID {annotation.id} submitted", disable_timer):
                                annotation.submit_annotations(
                                    result, session_state.user.id, conn)

                            time_elapsed = perf_counter() - start_task
                            logger.info(
                                f"Annotation submitted successfully [{time_elapsed:.4f}s]")

                        time_elapsed = perf_counter() - start_t
                        st.success(
                            "🎉 Successfully stored the uploaded annotations!")
                        logger.info("Successfully inserted all annotations for "
                                    f"{total_images} images into database. "
                                    f"Took {time_elapsed:.2f} seconds. "
                                    f"Average {time_elapsed / total_images:.4f}s per image")

                        with st.spinner("Updating project labels and editor configuration ..."):
                            if not is_updating:
                                default_labels = list(
                                    session_state.project.editor.get_labels())
                                logger.debug("Default labels found in template's "
                                             f"editor config: {default_labels}")
                                project_id = session_state.project.id
                                # get the unique labels from all the annotations
                                submitted_labels = session_state.project.get_existing_unique_labels()
                                # get new_labels to update editor config
                                new_labels = set(submitted_labels).difference(
                                    default_labels)
                                # get unwanted_labels to remove from editor config
                                unwanted_labels = set(default_labels).difference(
                                    new_labels)
                            else:
                                # get existing labels from editor_config to use to
                                # compare and update editor_config with new labels
                                existing_config_labels = list(
                                    session_state.project.editor.get_labels())
                                logger.debug("Existing labels found in "
                                             f"editor config: {existing_config_labels}")
                                # now the annotations will include new labels from the
                                # new uploaded annotations
                                existing_annotated_labels = session_state.project.get_existing_unique_labels()
                                new_labels = set(existing_annotated_labels).difference(
                                    existing_config_labels)
                            logger.info("Adding the new labels to editor config: "
                                        f"{new_labels}")

                            # update editor_config with the new labels from the uploaded annotations
                            for label in new_labels:
                                newChild = session_state.project.editor.create_label(
                                    'value', label)
                                logger.debug(
                                    f"newChild: {newChild.attributes.items()}")
                                session_state.project.editor.labels = session_state.project.editor.get_labels()
                            logger.info("All labels after updating: "
                                        f"{session_state.project.editor.labels}")

                            # default_labels = session_state.project.editor.get_default_template_labels()

                            # remove the unwanted default_labels came with the original
                            # editor_config template, but keep the ones from new_labels
                            if not is_updating and unwanted_labels:
                                for label in unwanted_labels:
                                    logger.debug(
                                        f"Removing label: {label}")
                                    session_state.project.editor.labels.remove(
                                        label)
                                    removedChild = session_state.project.editor.remove_label(
                                        'value', label)
                                    logger.debug(
                                        f"removedChild: {removedChild}")
                                session_state.project.editor.labels.sort()
                                logger.debug(f"After removing default labels: "
                                             f"{session_state.project.editor.labels}")

                            session_state.project.editor.update_editor_config()

                            session_state.project.refresh_project_details()

                        if error_imgs:
                            with st.expander("""NOTE: These images were unreadable and
                                skipped, but others are stored successfully in the
                                database. You may now go back to the Home page and 
                                enter the current project."""):
                                st.warning("  \n".join(error_imgs))
                        else:
                            st.success("""🎉 All images and annotations are successfully
                            stored in database. You may now go back to the Home page or 
                            enter the current project.""")

                        # clear out the "Submit" button to avoid further interactions
                        button_place.empty()
                else:
                    st.error(
                        f"Failed to stored **{session_state.new_dataset.name}** dataset information in database")
            else:
                st.error(
                    f"Failed to created **{session_state.new_dataset.name}** dataset")

            # remove the unneeded extracted archive dir contents
            with st.spinner("Removing the unwanted extracted files ..."):
                shutil.rmtree(session_state.new_dataset.archive_dir)
                logger.info(
                    "Removed temporary directory for extracted contents")

    # st.write("vars(session_state.new_dataset)")
    # st.write(vars(session_state.new_dataset))


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

        new_dataset(RELEASE=False, conn=conn)
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
