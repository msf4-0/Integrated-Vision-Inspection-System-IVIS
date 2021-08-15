"""
Title: Editor Manager
Date: 22/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

from os import stat
import sys
from pathlib import Path
from enum import IntEnum
from typing import List, Dict, Union
import xml.dom
from xml.dom import minidom
from datetime import datetime
import json
from PIL import Image
from base64 import b64encode, decode
from io import BytesIO
import streamlit as st
from streamlit import session_state as session_state

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
from core.utils.helper import get_mime
from data_manager.database_manager import db_no_fetch, init_connection, db_fetchone
from annotation.annotation_manager import annotation_types
from deployment.deployment_management import DEPLOYMENT_TYPE
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<
conn = init_connection(**st.secrets['postgres'])

# Editor table
# - id
# - name
# - editor_config
# - labels
# - project_id


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


# TAG_NAMES=
TAGNAMES = {
    "Image Classification": {"type": "Choices", "tag": "Choice"},
    "Object Detection with Bounding Boxes": {"type": "RectangleLabels", "tag": "Label"},
    "Semantic Segmentation with Polygons": {"type": "PolygonLabels", "tag": "Label"},
    "Semantic Segmentation with Masks": {"type": "BrushLabels", "tag": "Label"},
}


class BaseEditor:
    def __init__(self) -> None:

        # name is editor id for reference. Not the same as PK of DB
        self.id: int = None
        self.name: str = None
        self.editor_config: str = None
        self.labels: List = []
        self.project_id: Union[str, int] = None

    @staticmethod
    def get_annotation_tags(deployment_type):
        try:
            parent_tagname, child_tagname = TAGNAMES[deployment_type]['type'], TAGNAMES[deployment_type]['tag']
        except Exception as e:
            log_error(
                f"{e}: Could not retrieve tags. Deployment type '{deployment_type}' is not supported.")

        return parent_tagname, child_tagname


class NewEditor(BaseEditor):
    def __init__(self, random_generator) -> None:
        super().__init__()
        self.name: str = random_generator

    def init_editor(self) -> int:
        init_editor_SQL = """
                                    INSERT INTO public.editor (
                                        name,
                                        editor_config,
                                        project_id)
                                    VALUES (
                                        %s,
                                        %s,
                                        %s)
                                    RETURNING
                                        id;"""

        init_editor_vars = [self.name, self.editor_config, self.project_id]
        self.id = db_fetchone(init_editor_SQL, conn, init_editor_vars).id
        return self.id


class Editor(BaseEditor):
    def __init__(self, project_id, deployment_type) -> None:
        super().__init__()
        self.xml_doc: minidom.Document = None
        self.childNodes: minidom.Node = None
        self.project_id = project_id
        self.parent_tagname, self.child_tagname = self.get_annotation_tags(
            deployment_type)
        self.editor_config = self.load_raw_xml()
        self.xml_doc = self.load_xml(self.editor_config)
        self.query_editor_fields()

    def query_editor_fields(self):
        query_editor_fields_SQL = """
                SELECT                    
                    id,
                    name,
                    labels
                    
                FROM
                    public.editor 
                WHERE
                    project_id = %s
                """
        query_editor_fields_vars = [self.project_id]
        editor_fields = db_fetchone(
            query_editor_fields_SQL, conn, query_editor_fields_vars)
        if editor_fields:
            self.id, self.name, self.labels = editor_fields
        else:
            log_error(
                f"Editor for Project with ID: {self.project_id} does not exists in the database!!!")
        return editor_fields

    def load_raw_xml(self) -> str:
        """Load XML string from Database

        Returns:
            str: XML string
        """

        query_editor_SQL = """SELECT
                                editor_config
                            FROM
                                public.editor
                            WHERE
                                project_id = %s;"""

        query_editor_vars = [self.project_id]

        editor_config = db_fetchone(
            query_editor_SQL, conn, query_editor_vars)[0]

        self.editor_config = editor_config if editor_config else None
        if not editor_config:
            log_info(
                f"Editor config does not exists in the database for Project ID:{self.project_id}")

        return self.editor_config

    def load_xml(self, editor_config: str) -> minidom.Document:
        """Parse XML string into XML minidom.Document object

        Args:
            editor_config (str): XML string for database

        Returns:
            minidom.Document: XML minidom.Document object
        """
        if editor_config:
            xml_doc = minidom.parseString(editor_config)
            self.xml_doc = xml_doc
            return xml_doc
        else:
            log_error(f"Unable to parse string as XML object")

    @staticmethod
    def pretty_print(xml_doc: minidom.Document, encoding: str = 'utf-8'):
        """Pretty prints XML using minidom.Document.toprettyxml but removing additional whitespaces
            to give a compact and neater output. 

        Args:
            xml_doc (minidom.Document): XML object
            encoding (str, optional): Type of encoding of XML string. Defaults to 'utf-8'.

        Returns:
            str: XML string 
        """
        return '\n'.join([line for line in xml_doc.toprettyxml(indent='\t', encoding=encoding).decode('utf-8').split('\n') if line.strip()])

    def to_xml_string(self, pretty=False, encoding: str = 'utf-8', encoding_flag: bool = False) -> str:
        if pretty:
            xml_string = self.pretty_print(
                self.xml_doc, encoding=encoding)  # return string
            # xml_string = self.xml_doc.toprettyxml(
            #     encoding='utf8').decode('utf-8')
        else:
            xml_string = self.xml_doc.toxml(encoding=encoding).decode(encoding)

        if encoding_flag:
            xml_encoded_string = xml_string.encode(encoding)
            return xml_encoded_string

        else:
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
    def get_child(self, parent_tagName: str = None, child_tagName: str = None, attr: str = None, value: str = None) -> List:

        if (parent_tagName and child_tagName) is None:
            parent_tagName, child_tagName = self.parent_tagname, self.child_tagname

        parents = self.get_parents(parent_tagName, attr, value)
        elements = []
        for parent in parents:
            childs = parent.getElementsByTagName(
                child_tagName)  # list of child elements
            for child in childs:
                elements.append(child)
        self.childNodes = elements
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

    @staticmethod
    def get_labels_from_childNode(elements: List) -> List:
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
        # self.labels = labels
        return labels

    def get_labels(self, parent_tagName: str = None, child_tagName: str = None, attr: str = None, value: str = None):
        """Get labels from XML DOM using Parent and Child tags

        Args:
            parent_tagName (str): Annotation Tagname
            child_tagName (str): Annotation Child Tagname
            attr (str, optional): 'value' attribute of tag if specific label to be found . Defaults to None.
            value (str, optional): Value of 'value'. Defaults to None.

        Returns:
            [type]: [description]
        """
        if (parent_tagName and child_tagName) is None:
            parent_tagName, child_tagName = self.parent_tagname, self.child_tagname

        self.childNodes = self.get_child(
            parent_tagName=parent_tagName, child_tagName=child_tagName, attr=attr, value=value)
        self.labels = self.get_labels_from_childNode(self.childNodes)

        return self.labels

    def create_label(self, attr, value, parent_tagname=None, child_tagname=None):
        if (parent_tagname and child_tagname) is None:
            parent_tagname, child_tagname = self.parent_tagname, self.child_tagname
        nodeList = self.xml_doc.getElementsByTagName(
            parent_tagname)[0]  # 'RectangleLabels'

        new_label = self.xml_doc.createElement(child_tagname)  # 'Label'
        new_label.setAttribute(attr, value)  # value='<label-name>'

        # add new tag to parent childNodelist
        # <Label value="..." background="...">
        newChild = nodeList.appendChild(new_label)  # xml_doc will be updated

        # NOTE Update when submit button is pressed -> CALLBACK
        # serialise XML doc and Update database
        updated_editor_config_xml_string = self.to_xml_string(pretty=True)
        log_info(updated_editor_config_xml_string)
        # self.update_editor_config(updated_editor_config_xml_string)

        return newChild

    def edit_labels(self, attr: str, old_value: str, new_value: str, child_tagname: str = None):
        if child_tagname is None:
            child_tagname = self.child_tagname
        nodeList = self.xml_doc.getElementsByTagName(child_tagname)
        new_attributes = []
        for node in reversed(nodeList):
            if node.hasAttribute(attr) and node.getAttribute(attr) == old_value:
                node.setAttribute(attr, new_value)
                new_attributes.append((node.tagName, node.attributes.items()))
                log_info(
                    f"Label '{attr}:{old_value}' updated with attribute '{attr}:{new_value}'")
        if new_attributes:
            return new_attributes

    def remove_label(self, attr: str, value: str, child_tagname: str = None):
        if child_tagname is None:
            child_tagname = self.child_tagname
        nodeList = self.xml_doc.getElementsByTagName(child_tagname)
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
            # NOTE Update when submit button is pressed -> CALLBACK
            # updated_editor_config_xml_string = self.to_xml_string(pretty=True)
            # self.update_editor_config(updated_editor_config_xml_string)
            return removedChild

    def generate_labels_dict(self, deployment_type) -> dict:
        """  Generate labels dictionary to display on project dashboard
        {'Bounding Box':[List of labels],
            'Classification':[List of labels]
            } 

        Args:
            deployment_type ([type]): Type of Deep Learning deployment

        Returns:
            Dict: Dictionary of labels with Annotation Type
        """
        # get ID from enum member value
        # get annotation type
        # generate dict

        annotation_type = annotation_types[DEPLOYMENT_TYPE[deployment_type]]
        labels_dict = {annotation_type: self.labels}

        return labels_dict

    def update_editor_config(self, deployment_type: str):

        updated_editor_config_xml_string = self.to_xml_string(pretty=True)
        labels_dict = self.get_labels(deployment_type)
        labels_json = json.dumps(labels_dict)

        update_editor_config_SQL = """
                                    UPDATE
                                        public.editor
                                    SET
                                        editor_config = %s,
                                        labels = %s::JSONB
                                    WHERE
                                        project_id = %s;
                                            """
        update_editor_config_vars = [
            updated_editor_config_xml_string, labels_json, self.project_id]
        db_no_fetch(update_editor_config_SQL, conn, update_editor_config_vars)


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

    return data_url
