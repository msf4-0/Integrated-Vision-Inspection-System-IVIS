"""
Title: Editor Manager
Date: 22/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import json
import sys
import xml.dom
from base64 import b64encode, decode
from enum import Enum, IntEnum
from io import BytesIO
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple, Union
from xml.dom import minidom

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit import session_state as session_state

from core.utils.helper import create_dataframe
from data_export.label_studio_converter.converter import Converter

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

from annotation.annotation_management import annotation_types
from annotation.annotation_template import load_annotation_template
from core.utils.code_generator import get_random_string
from core.utils.log import logger  # logger
from data_manager.database_manager import (db_fetchone, db_no_fetch,
                                           init_connection)
from deployment.deployment_management import Deployment, DeploymentType
# >>>> User-defined Modules >>>>
from path_desc import chdir_root

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
    DeploymentType.Image_Classification: {"type": "Choices", "tag": "Choice"},
    DeploymentType.OD: {"type": "RectangleLabels", "tag": "Label"},
    DeploymentType.Instance: {"type": "PolygonLabels", "tag": "Label"},
    DeploymentType.Semantic: {"type": "BrushLabels", "tag": "Label"},
}


class Labels(NamedTuple):
    # store label details from self.labels_dict
    name: str
    type: str
    count: int
    percentile: float


class BaseEditor:
    def __init__(self) -> None:

        # name is editor id for reference. Not the same as PK of DB
        self.id: int = None
        self.name: str = None
        self.editor_config: str = None
        self.labels: List = []
        self.project_id: Union[str, int] = None
        self.deployment_type: Union[int, IntEnum] = None
        self.xml_doc: minidom.Document = None

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
            logger.error(f"Unable to parse string as XML object")

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

    def generate_labels_dict(self, deployment_type: IntEnum) -> dict:
        """  Generate labels dictionary to display on project dashboard
        {'Bounding Box':[List of labels],
            'Classification':[List of labels]
            } 

        Args:
            deployment_type ([type]): Type of Deep Learning deployment

        Returns:
            Dict: Dictionary of labels with Annotation Type
        """

        if not self.labels:
            # Case when Editor not exists in DB
            self.xml_doc: minidom.Document = self.load_xml(self.editor_config)
            self.parent_tagname, self.child_tagname = self.get_annotation_tags(
                deployment_type)
            self.labels = self.get_labels(
                self.parent_tagname, self.child_tagname)

        annotation_type = annotation_types[deployment_type]
        labels_dict = {annotation_type: self.labels}

        return labels_dict

    @staticmethod
    def get_annotation_tags(deployment_type):
        try:
            parent_tagname, child_tagname = TAGNAMES[deployment_type]['type'], TAGNAMES[deployment_type]['tag']
        except Exception as e:
            logger.error(
                f"{e}: Could not retrieve tags. Deployment type '{deployment_type}' is not supported.")

        return parent_tagname, child_tagname

    @staticmethod
    def get_editor_template(deployment_type_id: Union[int, IntEnum]) -> str:
        """Get editor template from config.yml

        Args:
            deployment_type_id (Union[int, IntEnum]): Deployment type id 

        Returns:
            str: Editor template XML string
        """

        editor_config = (load_annotation_template(
            deployment_type_id - 1))['config']

        return editor_config

    def convert_labels_dict_to_JSON(self):
        """Get labels from editor template XML and generate JSON based on the format:
            {'Bounding Box':[List of labels],
            'Classification':[List of labels]
            } 

        Returns:
            str: JSON string of labels
        """
        labels_dict = self.generate_labels_dict(self.deployment_type)
        labels_json = json.dumps(labels_dict)
        return labels_json

    def insert_editor_template(self) -> int:
        """Insert editor template and labels into Database
            - editor_config loaded from config.yml
            - labels obtained by iterating through XML doc

        Returns:
            int: Editor class id
        """
        self.deployment_type = Deployment.get_deployment_type(
            self.deployment_type)  # convert to IntEnum
        labels_json = self.convert_labels_dict_to_JSON()
        init_editor_SQL = """
                                    INSERT INTO public.editor (
                                        name,
                                        editor_config,
                                        project_id,
                                        labels)
                                    VALUES (
                                        %s,
                                        %s,
                                        %s,
                                        %s)
                                    RETURNING
                                        id;"""

        init_editor_vars = [self.name, self.editor_config,
                            self.project_id, labels_json]
        self.id = db_fetchone(init_editor_SQL, conn, init_editor_vars).id
        return self.id


class NewEditor(BaseEditor):
    def __init__(self, random_generator) -> None:
        super().__init__()
        self.name: str = random_generator
        self.parent_tagname: str = None
        self.child_tagname: str = None

    def init_editor(self, deployment_type: str) -> int:
        self.deployment_type = deployment_type
        self.id = self.insert_editor_template()

        return self.id


class Editor(BaseEditor):
    def __init__(self, project_id, deployment_type) -> None:
        super().__init__()
        self.project_id = project_id
        self.childNodes: minidom.Node = None
        # store query from 'labels' column
        self.labels_dict: Dict[List[str]] = {}
        self.deployment_type = Deployment.get_deployment_type(deployment_type)
        self.parent_tagname, self.child_tagname = self.get_annotation_tags(
            self.deployment_type)
        self.editor_config = self.load_raw_xml()
        self.xml_doc: minidom.Document = self.load_xml(self.editor_config)
        self.id, self.name, self.labels = self.query_editor_fields()
        self.labels_results: List = []  # store results from labels

    def editor_notfound_handler(self):
        """Insert editor template into database and store in class attributes
        - Editor config from config.yml
        - Editor.id from query return 
        """
        # generate temp name from random_generator length=8
        # get editor template
        # load into database with project id
        self.name = get_random_string(length=8)
        self.editor_config = self.get_editor_template(self.deployment_type)
        self.id = self.insert_editor_template()

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
            self.id, self.name, self.labels_dict = editor_fields
        else:
            logger.error(
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

        try:
            self.editor_config = (db_fetchone(
                query_editor_SQL, conn, query_editor_vars)[0])
            logger.info(f"Loaded editor from DB")
        except TypeError as e:
            logger.error(
                f"{e}: Editor config does not exists in the database for Project ID:{self.project_id}")
            self.editor_notfound_handler()
            logger.info(f"Loaded editor into DB with ID: {self.id}")

        return self.editor_config

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
                logger.info(
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
                    logger.error(error_msg)

            else:
                error_msg = f"Child node does not exist"
                logger.error(error_msg)

        if removedChild:
            # NOTE Update when submit button is pressed -> CALLBACK
            # updated_editor_config_xml_string = self.to_xml_string(pretty=True)
            # self.update_editor_config(updated_editor_config_xml_string)
            return removedChild

    def update_editor_config(self):

        self.editor_config = self.to_xml_string(pretty=True)
        labels_json = self.convert_labels_dict_to_JSON()

        update_editor_config_SQL = """
                                    UPDATE
                                        public.editor
                                    SET
                                        editor_config = %s,
                                        labels = %s::JSONB
                                    WHERE
                                        project_id = %s
                                    RETURNING id;
                                            """
        update_editor_config_vars = [
            self.editor_config, labels_json, self.project_id]
        query_return = db_fetchone(
            update_editor_config_SQL, conn, update_editor_config_vars)

        return query_return

    def get_labels_results(self):
        # Compatible with multiple annotation types
        try:
            self.labels_results = []
            for key, values in self.labels_dict.items():
                annotation_type = key
                for value in values:
                    self.labels_results.append(
                        Labels(value, annotation_type, None, None))
            logger.info(f"Getting Label Details (labels_results)")
        except TypeError as e:
            logger.error(f"{e}: Labels could not be found in 'labels' column")
            self.labels_results = []
            if self.labels:
                # Compatible with one annotation type
                # form dict from XML
                annotation_type = annotation_types[self.deployment_type]
                for value in self.labels:
                    self.labels_results.append(
                        Labels(value, annotation_type, None, None))
                logger.info(f"Getting Label Details (labels_dict)")

    def create_table_of_labels(self) -> pd.DataFrame:
        self.get_labels_results()
        # Create DataFrame
        column_names = ['Label Name', 'Annotation Type',
                        'Counts', 'Percentile (%)']
        df = create_dataframe(self.labels_results, column_names,
                              sort=True, sort_by='Annotation Type')
        df = df.fillna(0)
        return df

    @st.cache
    def get_labelstudio_converter(self):
        # initialize a Label Studio Converter to convert to specific output formats
        # the `editor.editor_config` contains the current project's config in XML string format
        #  but need to remove this line of encoding description text to work
        config_xml = self.editor_config.replace(
            r'<?xml version="1.0" encoding="utf-8"?>', '')
        converter = Converter(config=config_xml)
        return converter

    @st.cache
    def get_supported_format_info(
        self,
        converter: Converter = None
    ) -> Tuple[pd.DataFrame, Dict[str, Enum]]:
        """
        Get the supported formats from the converter. Then create a DataFrame from it,
        original column names are: title, description, link, tags. But we rename them to
        these columns: Format, Description, Reference Link, Tags. 

        Returns the DataFrame and also the Dict to convert from Format str to Format Enum,
        which is needed to use the converter.
        """
        if converter is None:
            converter = self.get_labelstudio_converter()
        df = pd.DataFrame(converter._FORMAT_INFO).T
        df.index = df.index.astype(str)

        supported_formats = converter.supported_formats
        df = df[df.index.isin(supported_formats)]
        # getting the mapping of index: title, i.e. Format Enum: Format str
        enum2str = df['title'].to_dict()
        str2enum = {v: k for k, v in enum2str.items()}

        # drop the index column containing the Format Enum
        df.reset_index(drop=True, inplace=True)
        df.columns = ['Format', 'Description',
                      'Reference Link', 'Tags']

        return df, str2enum


@st.cache
def load_sample_image():
    """Load Image and generate Data URL in base64 bytes

    Args:
        image (bytes-like): BytesIO object

    Returns:
        bytes: UTF-8 encoded base64 bytes
    """
    chdir_root()  # ./image_labelling
    logger.info("Loading Sample Image")
    sample_image = "resources/sample.jpg"
    with Image.open(sample_image) as img:
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='jpeg')

    bb = img_byte_arr.getvalue()
    b64code = b64encode(bb).decode('utf-8')
    data_url = 'data:image/jpeg;base64,' + b64code

    return data_url
