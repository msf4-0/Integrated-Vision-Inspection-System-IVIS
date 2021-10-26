"""
Title: Project Management
Date: 21/7/2021
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
import shutil
import sys
import json
from pathlib import Path
from collections import namedtuple
from typing import NamedTuple, Optional, Tuple, Union, List, Dict
from time import sleep, perf_counter
from enum import IntEnum
from glob import glob, iglob
import cv2
import numpy as np
import pandas as pd
from stqdm import stqdm
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state as session_state
import streamlit.components.v1 as components
from data_export.label_studio_converter.converter import Converter, Format, FormatNotSupportedError
# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

# >>>> User-defined Modules >>>>
from path_desc import PROJECT_DIR, chdir_root, DATASET_DIR
from core.utils.log import logger  # logger
from data_manager.database_manager import init_connection, db_fetchone, db_no_fetch, db_fetchall
from core.utils.file_handler import create_folder_if_not_exist, file_archive_handler
from core.utils.helper import get_directory_name, create_dataframe, dataframe2dict
from core.utils.form_manager import check_if_exists, check_if_field_empty, reset_page_attributes
from data_manager.dataset_management import Dataset, get_dataset_name_list
# Add CLI so can run Python script directly
from data_editor.editor_management import Editor
from annotation.annotation_management import Annotations, NewTask, Task, get_task_row

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration >>>>

# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


class ProjectPermission(IntEnum):
    ViewOnly = 0
    Edit = 1

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return ProjectPermission[s]
        except KeyError:
            raise ValueError()


class ProjectPagination(IntEnum):
    Dashboard = 0
    New = 1
    Existing = 2
    NewDataset = 3

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return ProjectPagination[s]
        except KeyError:
            raise ValueError()


class NewProjectPagination(IntEnum):
    Entry = 0
    NewDataset = 1
    EditorConfig = 2

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return NewProjectPagination[s]
        except KeyError:
            raise ValueError()


class ExistingProjectPagination(IntEnum):
    Dashboard = 0
    Labelling = 1
    Training = 2
    Models = 3
    Export = 4
    Settings = 5

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return ExistingProjectPagination[s]
        except KeyError:
            raise ValueError()
# <<<< Variable Declaration <<<<


class BaseProject:
    def __init__(self, project_id=None) -> None:
        self.id = project_id
        self.name: str = None
        self.desc: str = None
        self.project_path: Path = None
        self.deployment_id: Union[str, int] = None
        self.dataset_id: int = None
        self.training_id: int = None
        self.editor: str = None
        self.deployment_type: str = None
        self.dataset_chosen: List = []
        self.project = []  # keep?
        self.project_size: int = None  # Number of files
        self.dataset_list: Dict = {}
        self.image_name_list: List = []  # for image_labelling
        self.annotation_task_join = []  # for image_labelling

    # DEPRECATED

    @st.cache(ttl=60)
    def get_annotation_task_list(self):
        query_annotation_task_JOIN_SQL = """
            SELECT
                t.id AS "Task ID",
                t.name AS "Task Name",
                d.name AS "Dataset Name",
                t.is_labelled AS "Is Labelled",
                t.skipped AS "Skipped",
                a.updated_at AS "Date/Time"
            FROM
                annotations a
                INNER JOIN public.task t ON a.task_id = t.id
                INNER JOIN public.dataset d ON d.id = t.dataset_id
            WHERE
                t.project_id = %s
            ORDER BY
                d.name DESC;"""

        annotation_task_join, column_names = db_fetchall(query_annotation_task_JOIN_SQL, conn, [
            self.id], fetch_col_name=True)
        annotation_task_join_tmp = []
        if annotation_task_join:
            for annotation_task in annotation_task_join:

                # convert datetime with TZ to (2021-07-30 12:12:12) format
                converted_datetime = annotation_task.Date_Time.strftime(
                    '%Y-%m-%d %H:%M:%S')
                annotation_task = annotation_task._replace(
                    Date_Time=converted_datetime)
                annotation_task_join_tmp.append(annotation_task)

            self.annotation_task_join = annotation_task_join_tmp
        else:
            self.annotation_task_join = []

        return column_names

    @staticmethod
    def get_project_path(project_name: str) -> Path:
        """Get project path from project name

        Args:
            project_name (str): Project name

        Returns:
            Path: Path-like object for project path
        """
        directory_name = get_directory_name(
            project_name)  # change name to lowercase
        # join directory name with '-' dash
        project_path = PROJECT_DIR / str(directory_name)
        logger.info(f"Project Path: {str(project_path)}")

        return project_path

    # Wrapper for check_if_exists function from form_manager.py
    def check_if_exists(self, context: Dict, conn) -> bool:
        table = 'public.project'

        exists_flag = check_if_exists(
            table, context['column_name'], context['value'], conn)

        return exists_flag

    # Wrapper for check_if_exists function from form_manager.py
    def check_if_field_empty(self, context: Dict, field_placeholder: Dict, name_key: str):
        check_if_exists = self.check_if_exists
        empty_fields = check_if_field_empty(
            context, field_placeholder, name_key, check_if_exists)
        return empty_fields


class Project(BaseProject):
    def __init__(self, project_id: int) -> None:
        super().__init__(project_id)
        self.project_status = ProjectPagination.Existing  # flag for pagination
        self.datasets, self.column_names = self.query_project_dataset_list()
        self.dataset_dict = self.get_dataset_name_list()
        self.data_name_list = self.get_data_name_list()
        # `query_all_fields` creates `self.name`, `self.desc`, `self.deployment_type`, `self.deployment_id`
        self.query_all_fields()
        self.project_path = self.get_project_path(self.name)
        # Instantiate Editor class object
        self.editor: str = Editor(self.id, self.deployment_type)

    def download_tasks(self, *,
                       converter: Converter = None, export_format: str = None,
                       target_path: Path = None, return_original_path: bool = False,
                       return_target_path: bool = True) -> Union[None, Path]:
        """
        Download all the labeled tasks by archiving and moving them into the user's `Downloads` folder.
        Or you may also pass in a directory to the `target_path` parameter to move the file there.
        NOTE: If `return_original_path` is passed, this will only return the path to where the zipfile is created,
        and the zipfile will not be moved to the "Downloads" folder.
        """
        self.export_tasks(converter=converter, export_format=export_format)
        export_path = self.get_export_path()
        filename_no_ext = export_path.parent.name
        file_archive_handler(filename_no_ext, export_path, ".zip")
        zip_filename = f"{filename_no_ext}.zip"
        zip_filepath = export_path.parent / zip_filename

        if return_original_path:
            assert target_path is None, ("This will return the original path where the zipfile is created. "
                                         "`target_path` is not required.")
            # return the original zipfile path without moving it to target_path
            return zip_filepath

        if target_path is None:
            # default `target_path` is the "Downloads" folder
            target_path = Path.home() / "Downloads"

        shutil.move(zip_filepath, target_path)
        if return_target_path:
            return target_path

    def export_tasks(self, converter: Optional[Converter] = None,
                     export_format: Optional[str] = None,
                     for_training_id: Optional[int] = 0):
        """
        Export all annotated tasks into a specific format (e.g. Pascal VOC) and save to the dataset export directory.
        If `for_training_id` > 0, this will only export the annotated dataset associated with the `training_id`, to use for training.
        """
        logger.debug(
            f"Exporting labeled tasks for Project ID: {session_state.project.id}")

        output_dir = self.get_export_path()

        # - beware here I added removing the entire existing directory before proceeding
        if output_dir.exists():
            logger.warning(
                f"Removing existing exported directory: {output_dir}")
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        json_path = self.generate_label_json(
            for_training_id, output_dir=output_dir)

        if converter is None:
            converter = self.editor.get_labelstudio_converter()

        if export_format:
            export_format = Format.from_string(export_format)

        if self.deployment_type == "Image Classification":
            if for_training_id != 0:
                # using CSV format to get our image_paths for training
                export_format = Format.CSV

            # if it's not for training but using CSV format, then we copy the images
            #  into class folders to let the user download them
            if for_training_id == 0 and export_format == Format.CSV:
                logger.debug("Exporting in CSV format for the user")
                # training must use CSV file format for our implementation
                converter.convert_to_csv(
                    json_path,
                    output_dir=output_dir,
                    is_dir=False,
                )
                # the CSV file generated has these columns:
                # image, id, label, annotator, annotation_id, created_at, updated_at, lead_time.
                # The first col `image` contains the absolute paths to the images
                csv_path = output_dir / 'result.csv'
                df = pd.read_csv(csv_path)

                project_img_path = output_dir / "images"
                unique_labels = df['label'].unique().astype(str)
                for label in unique_labels:
                    class_path = project_img_path / label
                    logger.debug(
                        f"Creating folder for class '{label}' at {class_path}")
                    os.makedirs(class_path)

                def copy_images(image_path: Path, label: str):
                    class_path = project_img_path / label
                    shutil.copy2(image_path, class_path)

                logger.debug(
                    f"Copying images into each class folder in {project_img_path}")
                for row in df.values:
                    image_path = Path(row[0])   # first row for image_path
                    # getting the absolute path to the image
                    image_path = DATASET_DIR / image_path
                    label = str(row[2])         # third row for label name
                    copy_images(image_path, label)
                logger.info(
                    f"Image folders for each class {unique_labels} created successfully for Project ID {self.id}")
                # done and directly return back
                return

        elif self.deployment_type == "Object Detection with Bounding Boxes":
            if for_training_id != 0:
                # using Pascal VOC XML format for TensorFlow Object Detection API
                export_format = Format.VOC

        elif self.deployment_type == "Semantic Segmentation with Polygons":
            if for_training_id != 0:
                # using COCO JSON format for segmentation
                export_format = Format.COCO

        converter.convert(json_path, output_dir, export_format, is_dir=False)

        logger.debug(f"Exported tasks in format {export_format} for "
                     f"{self.deployment_type} for Project ID: {self.id}")

    def get_export_path(self) -> Path:
        """Get the path to the exported images and annotations"""
        project_path = self.get_project_path(self.name)
        output_dir = project_path / "export"
        return output_dir

    def generate_label_json(self, for_training_id: int = 0, output_dir: Path = None) -> Path:
        """
        Generate the output JSON with the format following Label Studio and returns the path to the file.
        Refer to 'resources/LS_annotations/bbox/labelstud_output.json' file as reference.
        If `for_training_id` is provided, then the JSON file is based on the annotations
        associated with the dataset used for the `training_id`.
        """
        all_annots, col_names = self.query_annotations(
            self.id, for_training_id)
        # the col_names can be changed if necessary
        df = pd.DataFrame(all_annots, columns=col_names)

        def get_image_path(image_path):
            # not using this for now because wants to use relative path instead
            # full_image_path = str((DATASET_DIR / image_path).resolve())
            return {"image": image_path}

        def create_annotations(result):
            return [{"result": result}]

        # create the format required to use Label Studio converter to export
        df['data'] = df['image_path'].apply(get_image_path)
        df['annotations'] = df['result'].apply(create_annotations)
        # drop the columns not necessary for converting
        df.drop(columns=['image_path', 'result'], inplace=True)

        # convert to json format to export to the project_path and use for conversion
        result = df.to_json(orient="records")
        if not output_dir:
            output_dir = self.get_export_path()
        else:
            os.makedirs(output_dir, exist_ok=True)
        json_path = output_dir / f"project-{self.id}.json"
        with open(json_path, "w") as f:
            parsed = json.loads(result)
            logger.debug(f"DUMPING TASK JSON to {json_path}")
            json.dump(parsed, f, indent=2)
        return json_path

    @staticmethod
    def get_existing_unique_labels(project_id: int) -> List[str]:
        """
        Extracting the unique label names used in existing annotations.
        Each 'result' value from the 'all_annots' is a list like this:
        ```
        "result": [
          {
            ...
            "value": {
              ...
              "label_key": ["Airplane", "Truck"]
            },
            ...
            "type": "label_key"
          }
        ```
        """
        # 'all_annots' is a list of dictionaries for each annotation
        all_annots, col_names = Project.query_annotations(
            project_id, return_dict=True)
        if not all_annots:
            # there is no existing annotation
            logger.error(f"No existing annotations for Project {project_id}")
            return []

        # the 'label_key' is different depending on the 'deployment_type'
        label_key = all_annots[0]['result'][0]['type']

        def get_label_names(result):
            return result[0]['value'][label_key]

        df = pd.DataFrame(all_annots, columns=col_names)
        df['label'] = df["result"].apply(get_label_names)

        # need to use `explode` method to turn each list of labels into individual rows
        unique_labels = sorted(df['label'].explode().unique())

        logger.info(
            f"Unique labels for Project ID {project_id}: {unique_labels}")
        return unique_labels

    @staticmethod
    def query_annotations(project_id: int, return_dict: bool = True, for_training_id: int = 0) -> Tuple[List[Dict], List]:
        """Query annotations for this project. If `for_training_id` is provided,
        then only the annotations associated with the `training_id` is queried."""
        if for_training_id > 0:
            sql_query = """
                SELECT a.id AS id,
                    a.result AS result,
                    -- flag of 'g' to match every pattern instead of only the first
                    CONCAT_WS('/', regexp_replace(trim(both from d.name), '\s+', '-', 'g'), t.name) AS image_path
                FROM annotations a
                        LEFT JOIN task t on a.id = t.annotation_id
                        LEFT JOIN training_dataset td on t.dataset_id = td.dataset_id
                        LEFT JOIN dataset d on td.dataset_id = d.id
                WHERE td.training_id = %s and a.project_id = %s
                ORDER BY id;
            """
            query_vars = [for_training_id, project_id]
            logger.info(f"""Querying annotations from database for Project ID: {project_id}
                        and Training ID: {for_training_id}""")
        else:
            sql_query = """
                    SELECT 
                        a.id AS id,
                        a.result AS result,
                        -- flag of 'g' to match every pattern instead of only the first
                        CONCAT_WS('/', regexp_replace(trim(both from d.name),'\s+','-', 'g'), t.name) AS image_path
                    FROM annotations a
                            LEFT JOIN task t on a.id = t.annotation_id
                            LEFT JOIN dataset d on t.dataset_id = d.id
                    WHERE a.project_id = %s
                    ORDER BY id;
            """
            query_vars = [project_id]
            logger.info(
                f"Querying annotations from database for Project ID: {project_id}")

        try:
            all_annots, column_names = db_fetchall(
                sql_query, conn, query_vars, fetch_col_name=True, return_dict=return_dict)
            logger.debug(f"Total annotations: {len(all_annots)}")

        except Exception as e:
            logger.error(f"{e}: No annotation found for Project {project_id} ")
            all_annots = []
            column_names = []

        return all_annots, column_names

    def query_all_fields(self) -> NamedTuple:
        query_all_field_SQL = """
                            SELECT
                                
                                p.name,
                                description,
                                dt.name as deployment_type,
                                deployment_id                              
                                
                            FROM
                                public.project p
                                LEFT JOIN deployment_type dt ON dt.id = p.deployment_id
                            WHERE
                                p.id = %s;
                            """
        query_all_field_vars = [self.id]
        project_field = db_fetchone(
            query_all_field_SQL, conn, query_all_field_vars)
        if project_field:
            self.name, self.desc, self.deployment_type, self.deployment_id = project_field
        else:
            logger.error(
                f"Project with ID: {self.id} does not exists in the database!!!")
        return project_field

    @st.cache(ttl=60)
    def query_project_dataset_list(self) -> List:
        query_project_dataset_SQL = """
                                SELECT
                                    d.id AS "ID",
                                    d.name AS "Name",
                                    d.dataset_size AS "Dataset Size",
                                    (SELECT ft.name AS "File Type" from public.filetype ft where ft.id = d.filetype_id),
                                    pd.updated_at AS "Date/Time"                                    
                                FROM
                                    public.project_dataset pd
                                    LEFT JOIN public.dataset d ON d.id = pd.dataset_id
                                WHERE
                                    pd.project_id = %s
                                ORDER BY d.id ASC;
                                    """
        query_project_dataset_vars = [self.id]
        project_datasets, column_names = db_fetchall(
            query_project_dataset_SQL, conn, query_project_dataset_vars, fetch_col_name=True)

        logger.info(
            "Querying list of dataset attached to project from database......")
        project_dataset_tmp = []
        if project_datasets:
            for dataset in project_datasets:
                # convert datetime with TZ to (2021-07-30 12:12:12) format
                converted_datetime = dataset.Date_Time.strftime(
                    '%Y-%m-%d %H:%M:%S')
                dataset = dataset._replace(
                    Date_Time=converted_datetime)
                project_dataset_tmp.append(dataset)

            # self.datasets = project_dataset_tmp
        else:
            project_dataset_tmp = []

        return project_dataset_tmp, column_names

    def get_dataset_name_list(self):
        """Generate Dictionary of namedtuple

        Args:
            project_dataset_list (List[namedtuple]): Query from database

        Returns:
            Dict: Dictionary of namedtuple
        """
        project_dataset_dict = {}
        # if self.datasets:
        #     for dataset in self.datasets:
        #         # DEPRECATED -> dataset info can be accessed through namedtuple of dataset_dict
        #         # dataset_name_list[dataset.Name] = dataset.ID
        #         project_dataset_dict[dataset.Name] = dataset
        project_dataset_dict = get_dataset_name_list(self.datasets)  # UPDATED
        logger.info("Generating list of project dataset names and ID......")

        return project_dataset_dict

    def refresh_project_details(self):
        """Redundant function to update project attributes
        """
        self.datasets, self.column_names = self.query_project_dataset_list()
        self.dataset_dict = self.get_dataset_name_list()
        self.data_name_list = self.get_data_name_list()
        self.query_all_fields()

    # DEPRECATED

    @st.cache
    def load_dataset(self) -> List:
        """Loads data from the dataset directory and stored as Numpy arrays using OpenCV. 

        Returns:
            List: Dictionary with dataname as key to the respective numpy object
        """
        #args: self.datasets
        # return: dataset_name_list and image list

        if self.datasets:
            start_time = perf_counter()
            dataset_name_list = []
            dataset_list = {}
            data_name_list = {}
            for d in self.datasets:  # dataset loop
                dataset_name_list.append(d[1])  # get name
                dataset_path = d[4]
                logger.debug(f"Dataset {d[0]}:{dataset_path}")
                dataset_path = dataset_path + "/*"
                image_list = {}
                # data_name_tmp = []
                # image loop with sorted directories
                for image_path in iglob(dataset_path):
                    image = cv2.imread(image_path)  # get data url
                    image_name = (Path(image_path).name)
                    image_list[image_name] = image
                    # data_name_tmp.append(image_name)

                dataset_list[d[1]] = image_list
                # data_name_list[d[1]] = data_name_tmp
            self.dataset_name_list = dataset_name_list
            self.dataset_list = dataset_list
            # self.data_name_list = data_name_list
            end_time = perf_counter()
            logger.debug(end_time - start_time)

            return dataset_list

    @st.cache
    def get_data_name_list(self):
        """Obtain list of data in the dataset 
            - Iterative glob through the dataset directory
            - Obtain filename using pathlib.Path(<'filepath/*'>).name

        Returns:
            Dict[dict]: Dataset name as key to a List of data in the dataset directory
        """
        if self.datasets:
            data_name_list = {}
            for d in self.datasets:
                # data_name_tmp = []
                # dataset_path = Dataset.get_dataset_path(d.Name)
                # dataset_path = dataset_path / "./*"
                # # for data_path in iglob(dataset_path):
                # #     data_name = Path(data_path).name
                # #     data_name_tmp.append(data_name)

                # data_name_tmp = [Path(data_path).name
                #                  for data_path in iglob(str(dataset_path))]  # UPDATED with List comprehension
                data_name_tmp = get_single_data_name_list(d.Name)

                data_name_list[d.Name] = sorted(data_name_tmp)

            return data_name_list

    # *************************************************************************************************************************
    # TODO #81 Add reset to project page *************************************************************************************

    @staticmethod
    def reset_project_page():
        """Method to reset all widgets and attributes in the Project Page when changing pages
        """

        project_attributes = ["all_project_table", "project", "editor",
                              "labelling_pagination", "existing_project_pagination"]

        reset_page_attributes(project_attributes)
    # TODO #81 Add reset to project page *************************************************************************************
    # *************************************************************************************************************************


class NewProject(BaseProject):
    def __init__(self, project_id) -> None:
        # init BaseDataset -> Temporary dataset ID from random gen
        super().__init__(project_id)
        self.project_total_filesize = 0  # in byte-size
        self.has_submitted = False
        self.project_status = ProjectPagination.New  # flag for pagination

    def query_deployment_id(self) -> int:
        query_id_SQL = """
                        SELECT
                            id
                        FROM
                            public.deployment_type
                        WHERE
                            name = %s;
                        """
        if self.deployment_type is not None and self.deployment_type != '':

            self.deployment_id = db_fetchone(
                query_id_SQL, conn, [self.deployment_type])[0]
        else:
            self.deployment_id = None

    def insert_new_project_task(self, dataset_name: str, dataset_id: int):
        """Create New Task for Project
            - Insert into 'task' table

        Args:
            dataset_name (str): Dataset Name
            dataset_id (int): Dataset ID
        """
        data_name_list = get_single_data_name_list(dataset_name)
        if len(data_name_list):
            logger.info(f"Inserting task into DB........")
            for data in stqdm(data_name_list, unit='data', st_container=st.sidebar, desc='Creating task in database'):

                # >>>> Insert new task from NewTask class method
                task_id = NewTask.insert_new_task(
                    data, self.id, dataset_id)
                logger.info(f"Loaded task {task_id} into DB for data: {data}")

    def insert_project_dataset(self, dataset_dict: Dict[str, namedtuple]):
        insert_project_dataset_SQL = """
                                        INSERT INTO public.project_dataset (
                                            project_id,
                                            dataset_id)
                                        VALUES (
                                            %s,
                                            %s);"""

        for dataset in stqdm(self.dataset_chosen,
                             unit='dataset',
                             st_container=st.sidebar,
                             desc="Attaching dataset to project"):
            dataset_id = dataset_dict[dataset].ID
            dataset_name = dataset_dict[dataset].Name

            insert_project_dataset_vars = [self.id, dataset_id]
            db_no_fetch(insert_project_dataset_SQL, conn,
                        insert_project_dataset_vars)

            # NEED TO ADD INSERT TASK
            # get data name list
            # loop data and add task
            self.insert_new_project_task(dataset_name, dataset_id)

    def insert_project(self, dataset_dict: Dict[str, namedtuple] = None):
        """Insert project into database. If `dataset_dict` is provied,
        then also insert the project_dataset based on `self.dataset_chosen`."""
        insert_project_SQL = """
                                INSERT INTO public.project (
                                    name,
                                    description,                                                                   
                                    deployment_id)
                                VALUES (
                                    %s,
                                    %s,
                                    (SELECT dt.id FROM public.deployment_type dt where dt.name = %s))
                                RETURNING id;
                                
                            """
        insert_project_vars = [self.name, self.desc, self.deployment_type]

        # Query returns Project ID from table insertion
        self.id = db_fetchone(
            insert_project_SQL, conn, insert_project_vars).id

        if dataset_dict is not None:
            # only insert project_dataset when it is provided
            self.insert_project_dataset(dataset_dict)

        return self.id

    def initialise_project(self, dataset_dict: Dict[str, namedtuple] = None):

        # directory_name = get_directory_name(self.name)
        # self.project_path = PROJECT_DIR / str(directory_name)

        self.project_path = self.get_project_path(self.name)  # UPDATED

        create_folder_if_not_exist(self.project_path)

        logger.info(
            f"Successfully created **{self.name}** project at {str(self.project_path)}")

        if self.insert_project(dataset_dict):

            logger.info(
                f"Successfully stored **{self.name}** project information in database")
            return True

        else:
            logger.error(
                f"Failed to stored **{self.name}** project information in database")
            return False

    @staticmethod
    def reset_new_project_page():
        """Method to reset all widgets and attributes in the New Project Page when changing pages
        """

        new_project_attributes = ["new_project", "new_editor", "new_project_name",
                                  "new_project_desc", "annotation_type", "new_project_dataset_page", "new_project_dataset_chosen"]

        reset_page_attributes(new_project_attributes)


# ******************** QUERY ALL PROJECTS **************************************
# NOTE: You should not cache this, otherwise the brand new project created
#   will not be reflected on the `data_table` immediately
# @st.cache(ttl=60)
def query_all_projects(return_dict: bool = False, for_data_table: bool = False) -> Union[List[namedtuple], List[dict]]:
    """Return values for all project

    Args:
        return_dict (bool, optional): True if results to be in Python Dictionary, else collections.namedtuple. Defaults to False.

    Returns:
        List[NamedTuple]: [description]
    """
    ID_string = "id" if for_data_table else "ID"
    query_all_projects_SQL = f"""
                                SELECT
                                    p.id as \"{ID_string}\",
                                    p.name as "Name",
                                    description as "Description",
                                    dt.name as "Deployment Type",
                                    p.updated_at as "Date/Time"
                                    
                                FROM
                                    public.project p
                                    LEFT JOIN deployment_type dt ON dt.id = p.deployment_id
                                ORDER BY "Date/Time" DESC;
                            """
    projects, column_names = db_fetchall(
        query_all_projects_SQL, conn, fetch_col_name=True, return_dict=return_dict)

    logger.info(f"Querying projects from database")
    project_tmp = []
    if projects:
        for project in projects:
            # convert datetime with TZ to (2021-07-30 12:12:12) format
            if return_dict:
                converted_datetime = project["Date/Time"].strftime(
                    '%Y-%m-%d %H:%M:%S')
                project["Date/Time"] = converted_datetime
            else:
                converted_datetime = project.Date_Time.strftime(
                    '%Y-%m-%d %H:%M:%S')

                project = project._replace(
                    Date_Time=converted_datetime)
            project_tmp.append(project)

    # self.dataset_list = dataset_tmp

    else:
        project_tmp = []

    return project_tmp, column_names


# *********************NEW PROJECT PAGE NAVIGATOR ********************************************
def new_project_nav(color, textColor):
    textColor = textColor
    html_string = f'''
    <style>
      .div1 {{
        display: flex;
        justify-content: space-evenly;

        padding: 10px;
        width: 30%;
        margin-left: auto;
        margin-right: auto;
      }}
      .div2 {{
      color:{textColor};
        border-radius: 25px 0px 0px 25px;
        border-style: solid;
        border-color: {color[0].border};
        background-color: {color[0].background};
        padding: 10px;
        width: 100%;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
      }}
      .div3 {{
      color:{textColor};
        border-radius: 0px 25px 25px 0px;
        border-style: solid;
        border-color: {color[1].border};
        background-color: {color[1].background};
        padding: 10px;
        width: 100%;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
      }}
    </style>
    
      <div class="div1">
        <div class="div2">New Project Info</div>
        <div class="div3">Editor Setup</div>
      </div>
    
  '''
    components.html(html_string, height=100)

# >>>> CREATE PROJECT >>>>


def get_single_data_name_list(dataset_name: str) -> List:
    """Get a List of data for a single dataset

    Args:
        dataset_name (str): Name of dataset

    Returns:
        List: List of data from dataset
    """
    data_name_list = []
    dataset_path = Dataset.get_dataset_path(dataset_name)
    dataset_path = dataset_path / "./*"
    # for data_path in iglob(dataset_path):
    #     data_name = Path(data_path).name
    #     data_name_tmp.append(data_name)

    data_name_list = [Path(data_path).name
                      for data_path in iglob(str(dataset_path))]  # UPDATED with List comprehension

    return data_name_list


def main():
    print("Hi")


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
