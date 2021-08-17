"""
Title: Project Management
Date: 21/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
from typing import NamedTuple, Union, List, Dict
from time import sleep, perf_counter
from enum import IntEnum
from glob import glob, iglob
import cv2
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state as session_state
import streamlit.components.v1 as components
# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

# >>>> User-defined Modules >>>>
from path_desc import chdir_root, PROJECT_DIR
from core.utils.log import log_info, log_error  # logger
from data_manager.database_manager import init_connection, db_fetchone, db_no_fetch, db_fetchall
from core.utils.file_handler import create_folder_if_not_exist
from core.utils.helper import NavColor, get_directory_name
from core.utils.form_manager import check_if_exists, check_if_field_empty, reset_page_attributes
from data_manager.dataset_management import Dataset, get_dataset_name_list
# Add CLI so can run Python script directly
from data_editor.editor_management import Editor

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration >>>>

# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


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

    @st.cache(ttl=600)
    def query_all_projects(self) -> List[NamedTuple]:
        query_all_projects_SQL = """
                                    SELECT
                                        p.id as "ID",
                                        p.name as "Name",
                                        description as "Description",
                                        dt.name as "Deployment Type",
                                        project_path
                                    FROM
                                        public.project p
                                        LEFT JOIN deployment_type dt ON dt.id = p.deployment_id;
                                """
        projects = db_fetchall(
            query_all_projects_SQL, conn)

        return projects

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
        log_info(f"Project Path: {str(project_path)}")

        return project_path


class Project(BaseProject):
    def __init__(self, project_id: int) -> None:
        super().__init__(project_id)
        self.project_status = ProjectPagination.Existing  # flag for pagination
        self.datasets, self.column_names = self.query_project_dataset_list()
        self.dataset_dict = self.get_dataset_name_list()
        self.data_name_list = self.get_data_name_list()
        self.query_all_fields()
        self.project_path = self.get_project_path(self.name)
        # Instantiate Editor class object
        self.editor: str = Editor(self.id, self.deployment_type)

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
            log_error(
                f"Project with ID: {self.id} does not exists in the database!!!")
        return project_field

    def query_project_dataset_list(self) -> List:
        query_project_dataset_SQL = """
                                SELECT
                                    d.id AS "ID",
                                    d.name AS "Name",
                                    d.dataset_size AS "Dataset Size",
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

        log_info("Querying list of dataset attached to project from database......")
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
        log_info("Generating list of project dataset names and ID......")

        return project_dataset_dict

    def refresh_project_details(self):
        """Redundant function to update project attributes
        """
        self.datasets, self.column_names = self.query_project_dataset_list()
        self.dataset_dict = self.get_dataset_name_list()
        self.data_name_list = self.get_data_name_list()
        self.query_all_fields()

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
                log_info(f"Dataset {d[0]}:{dataset_path}")
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
            log_info(end_time - start_time)

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
                data_name_tmp = []
                dataset_path = Dataset.get_dataset_path(d.Name)
                dataset_path = dataset_path / "./*"
                # for data_path in iglob(dataset_path):
                #     data_name = Path(data_path).name
                #     data_name_tmp.append(data_name)

                data_name_tmp = [Path(data_path).name
                                 for data_path in iglob(str(dataset_path))]  # UPDATED with List comprehension

                data_name_list[d.Name] = sorted(data_name_tmp)

            return data_name_list


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

    # Wrapper for check_if_exists function from form_manager.py
    def check_if_exists(self, context: Dict, conn) -> bool:
        table = 'public.project'

        exists_flag = check_if_exists(
            table, context['column_name'], context['value'], conn)

        return exists_flag

    # Wrapper for check_if_exists function from form_manager.py
    def check_if_field_empty(self, context: Dict, field_placeholder):
        check_if_exists = self.check_if_exists
        empty_fields = check_if_field_empty(
            context, field_placeholder, check_if_exists)
        return empty_fields

    def insert_project(self, dataset_dict: Dict):
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
        self.id = db_fetchone(
            insert_project_SQL, conn, insert_project_vars).id

        insert_project_dataset_SQL = """
                                        INSERT INTO public.project_dataset (
                                            project_id,
                                            dataset_id)
                                        VALUES (
                                            %s,
                                            %s);"""
        for dataset in self.dataset_chosen:
            dataset_id = dataset_dict[dataset].ID
            insert_project_dataset_vars = [self.id, dataset_id]
            db_no_fetch(insert_project_dataset_SQL, conn,
                        insert_project_dataset_vars)
        return self.id

    def initialise_project(self, dataset_dict):

        # directory_name = get_directory_name(self.name)
        # self.project_path = PROJECT_DIR / str(directory_name)

        self.project_path = self.get_project_path(self.name)  # UPDATED

        create_folder_if_not_exist(self.project_path)

        log_info(
            f"Successfully created **{self.name}** project at {str(self.project_path)}")

        if self.insert_project(dataset_dict):

            log_info(
                f"Successfully stored **{self.name}** project information in database")
            return True

        else:
            log_error(
                f"Failed to stored **{self.name}** project information in database")
            return False

    @staticmethod
    def reset_new_project_page():
        """Method to reset all widgets and attributes in the New Project Page when changing pages
        """

        new_project_attributes = ["new_project", "new_editor", "new_project_name",
                                  "new_project_desc", "annotation_type", "new_project_dataset_page", "new_project_dataset_chosen"]

        reset_page_attributes(new_project_attributes)


# *********************NEW PROJECT PAGE NAVIGATOR ********************************************
def new_project_nav(color,textColor):
    
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


def main():
    print("Hi")


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
