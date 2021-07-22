"""
Title: Editor Manager
Date: 22/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

from os import name
import sys
from pathlib import Path
from typing import List, Dict, Union
import xml.dom
from xml.dom import minidom
from datetime import datetime
import psycopg2
import json
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as SessionState
from streamlit import error_util

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
from data_manager.database_manager import db_no_fetch, init_connection, db_fetchone

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<
conn = init_connection(**st.secrets['postgres'])

# Editor table
# - id
# - name
# - editor_config
# - labels
# - project_id


class BaseEditor:
    def __init__(self, random_generator) -> None:

        # name is editor id for reference. Not the same as PK of DB
        self.name: str = random_generator
        self.editor_config: str = None
        self.labels: List = []
        self.project_id: Union[str, int] = None


class Editor(BaseEditor):
    def __init__(self, random_generator) -> None:
        super().__init__(random_generator)
        self.xml_doc: minidom.Document = None
        self.childNodes: minidom.Node = None

    def load_xml(self, editor_config: str) -> minidom.Document:
        if editor_config:
            xml_doc = minidom.parseString(editor_config)
            self.xml_doc = xml_doc
            return xml_doc
        else:
            pass

    def to_xml_string(self, pretty=False) -> str:
        if pretty:
            xml_string = self.xml_doc.toprettyxml(
                encoding='utf8').decode('utf-8')
        else:
            xml_string = self.xml_doc.toxml(encoding='utf8').decode('utf-8')

        return xml_string

    # get Nodelist of parent tag
    def get_parents(self, parent_tagName: str, attr: str = None, value: str = None) -> List:
        if self.xml_doc:
            if attr and value:
                pass
            else:
                parents = self.xml_doc.getElementsByTagName(parent_tagName)
            return parents

    # to get list of labels
    def get_child(self, parent_tagName: str, child_tagName: str, attr: str = None, value: str = None) -> List:
        parents = self.get_parents(parent_tagName, attr, value)
        elements = []
        for parent in parents:
            childs = parent.getElementsByTagName(
                child_tagName)  # list of child elements
            for child in childs:
                elements.append(child)

        return elements

    def get_tagname_attributes(self, elements: List) -> List:
        '''
        element.attributes.items() -> give a list of tuples of attributes
        [('value', 'Hello'), ('background', 'blue')]
        [('value', 'World'), ('background', 'pink')]
        [('value', 'Hello'), ('background', 'blue')]
        [('value', 'World'), ('background', 'pink')]
        '''
        tagName_attributes = []
        for element in elements:
            tagName_attributes.append(
                (element.tagName, element.attributes.items()))

        return tagName_attributes

    def get_labels(self, elements: List) -> List:
        # assume only one type of annotation type
        labels = []
        for element in elements:  # for 'value' attrib ONLY
            if element.hasAttribute('value'):
                labels.append(element.getAttribute('value'))

        # TODO: add option for background
        # element.attributes.items() -> give a list of tuples of attributes
        # [('value', 'Hello'), ('background', 'blue')]
        # [('value', 'World'), ('background', 'pink')]
        # [('value', 'Hello'), ('background', 'blue')]
        # [('value', 'World'), ('background', 'pink')]
        return labels

    def create_label(self, parent, tagname, attr, value):
        nodeList = self.xml_doc.getElementsByTagName(
            parent)[0]  # 'RectangleLabels'

        new_label = self.xml_doc.createElement(tagname)  # 'Label'
        new_label.setAttribute(attr, value)  # value='<label-name>'

        # add new tag to parent childNodelist
        newChild = nodeList.appendChild(new_label)
        return newChild

    def edit_labels(self, tagName: str, attr: str, old_value: str, new_value: str):
        nodeList = self.xml_doc.getElementsByTagName(tagName)
        new_attributes = []
        for node in reversed(nodeList):
            if node.hasAttribute(attr) and node.getAttribute(attr) == old_value:
                node.setAttribute(attr, new_value)
                new_attributes.append((node.tagName, node.attributes.items()))
                log_info(
                    f"Label '{attr}:{old_value}' updated with attribute '{attr}:{new_value}'")
        if new_attributes:
            return new_attributes

    def remove_labels(self, tagName: str, attr: str, value: str):
        nodeList = self.xml_doc.getElementsByTagName(tagName)
        removedChild = []
        for node in reversed(nodeList):
            if node.hasAttribute(attr) and node.getAttribute(attr) == value:
                parent = node.parentNode
                try:
                    removedChild.append(parent.removeChild(node))

                except ValueError as e:
                    error_msg = f"{e}: Child node does not exist"
                    log_error(error_msg)

            else:
                error_msg = f"Child node does not exist"
                log_error(error_msg)
        if removedChild:
            return removedChild


# ************************************************* OLD *************************************************


class Results:
    def __init__(self, from_name, to_name, type, value) -> None:
        self.from_name: str = from_name
        self.to_name: str = to_name
        self.type: str = type
        self.value: List[Dict] = value


class Annotations:
    def __init__(self) -> None:
        self.id: int = 0
        self.completed_by: Dict = {}  # user_id, email, first_name, last_name
        self.was_cancelled: bool = False
        self.ground_truth: bool = True
        self.created_at: datetime = datetime.now().astimezone()
        self.updated_at: datetime = datetime.now().astimezone()
        self.lead_time: float = 0
        self.task: int = 0
        self.results = Results()  # or Dict?


def submit_annotations(results: Dict, project_id: int, users_id: int, task_id: int, annotation_id: int, is_labelled: bool = True, conn=conn) -> int:
    """ Submit results for new annotations

    Args:
        results (Dict): [description]
        project_id (int): [description]
        users_id (int): [description]
        task_id (int): [description]
        annotation_id (int): [description]
        is_labelled (bool, optional): [description]. Defaults to True.
        conn (psycopg2 connection object, optional): [description]. Defaults to conn.

    Returns:
        [type]: [description]
    """

    # TODO is it neccessary to have annotation type id?
    insert_annotations_SQL = """
                                INSERT INTO public.annotations (
                                    results,
                                    project_id,
                                    users_id,
                                    task_id)
                                VALUES (
                                    %s::jsonb,
                                    %s,
                                    %s, 
                                    %s) 
                                RETURNING id;
                            """, [json.dumps(results), project_id, users_id, task_id]
    annotation_id = db_fetchone(insert_annotations_SQL, conn)

    update_task_SQL = """
                        UPDATE
                            public.task
                        SET
                            (annotation_id = %s),
                            (is_labelled = %s)
                        WHERE
                            id = %s;
                    """
    context = [annotation_id, is_labelled, task_id]
    db_no_fetch(update_task_SQL, conn, context)

    return annotation_id


def update_annotations(results: Dict, users_id: int, annotation_id: int, conn=conn) -> tuple:
    """Update results for new annotations

    Args:
        results (Dict): [description]
        users_id (int): [description]
        annotation_id (int): [description]
        conn (psycopg2 connection object, optional): [description]. Defaults to conn.

    Returns:
        tuple: [description]
    """

    # TODO is it neccessary to have annotation type id?
    update_annotations_SQL = """
                                UPDATE
                                    public.annotations
                                SET
                                    (results = %s::jsonb),
                                    (users_id = %s)
                                WHERE
                                    id = %s
                                RETURNING *;
                            """, [json.dumps(results), users_id]
    updated_annotation_return = db_fetchone(update_annotations_SQL, conn)

    return updated_annotation_return


def skip_task(task_id: int, skipped: bool) -> tuple:
    """Skip task

    Args:
        task_id (int): [description]
        skipped (bool): [description]

    Returns:
        tuple: [description]
    """
    skip_task_SQL = """
                    UPDATE
                        public.task
                    SET
                        (skipped = %s)
                    WHERE
                        id = %s;
                """, [skipped, task_id]

    skipped_task_return = db_fetchone(skip_task_SQL, conn)

    return skipped_task_return


def delete_annotation(annotation_id: int) -> tuple:
    """Delete annotations

    Args:
        annotation_id (int): [description]

    Returns:
        tuple: [description]
    """
    delete_annotations_SQL = """
                            DELETE FROM public.annotation
                            WHERE id = %s
                            RETURNING *;
                            """, [annotation_id]

    delete_annotation_return = db_fetchone(delete_annotations_SQL, conn)

    return delete_annotation_return
