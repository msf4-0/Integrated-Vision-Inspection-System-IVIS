# %% [markdown]
# # Test LS Converter

# %%

from pathlib import Path
import sys
import os
import io
from glob import glob, iglob
import json
import yaml
import shutil
from zipfile import ZipFile
import tarfile
import urllib
from collections import Mapping, defaultdict

from typing import Union, List, Dict

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[3]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

from data_export.label_studio_converter.converter import Converter

from lxml import etree
from collections import defaultdict
import logging
from copy import deepcopy
from enum import Enum
import json
logger = logging.getLogger(__name__)
_LABEL_TAGS = {'Label', 'Choice'}
_NOT_CONTROL_TAGS = {'Filter', }



def parse_config(config_string):
    """
    :param config_string: Label config string
    :return: structured config of the form:
    {
        "<ControlTag>.name": {
            "type": "ControlTag",
            "to_name": ["<ObjectTag1>.name", "<ObjectTag2>.name"],
            "inputs: [
                {"type": "ObjectTag1", "value": "<ObjectTag1>.value"},
                {"type": "ObjectTag2", "value": "<ObjectTag2>.value"}
            ],
            "labels": ["Label1", "Label2", "Label3"] // taken from "alias" if exists or "value"
    }
    """
    if not config_string:
        return {}

    def _is_input_tag(tag):
        return tag.attrib.get('name') and tag.attrib.get('value')

    def _is_output_tag(tag):
        return tag.attrib.get('name') and tag.attrib.get('toName') and tag.tag not in _NOT_CONTROL_TAGS

    def _get_parent_output_tag_name(tag, outputs):
        # Find parental <Choices> tag for nested tags like <Choices><View><View><Choice>...
        parent = tag
        while True:
            parent = parent.getparent()
            if parent is None:
                return
            name = parent.attrib.get('name')
            if name in outputs:
                return name

    xml_tree = etree.fromstring(config_string)

    inputs, outputs, labels = {}, {}, defaultdict(dict)
    for tag in xml_tree.iter():
        if _is_output_tag(tag):
            tag_info = {'type': tag.tag,
                        'to_name': tag.attrib['toName'].split(',')}
            # Grab conditionals if any
            conditionals = {}
            if tag.attrib.get('perRegion') == 'true':
                if tag.attrib.get('whenTagName'):
                    conditionals = {'type': 'tag',
                                    'name': tag.attrib['whenTagName']}
                elif tag.attrib.get('whenLabelValue'):
                    conditionals = {'type': 'label',
                                    'name': tag.attrib['whenLabelValue']}
                elif tag.attrib.get('whenChoiceValue'):
                    conditionals = {'type': 'choice',
                                    'name': tag.attrib['whenChoiceValue']}
            if conditionals:
                tag_info['conditionals'] = conditionals
            outputs[tag.attrib['name']] = tag_info
        elif _is_input_tag(tag):
            inputs[tag.attrib['name']] = {
                'type': tag.tag, 'value': tag.attrib['value'].lstrip('$')}
        if tag.tag not in _LABEL_TAGS:
            continue
        parent_name = _get_parent_output_tag_name(tag, outputs)
        if parent_name is not None:
            actual_value = tag.attrib.get('alias') or tag.attrib.get('value')
            if not actual_value:
                logger.debug(
                    'Inspecting tag {tag_name}... found no "value" or "alias" attributes.'.format(
                        tag_name=etree.tostring(tag, encoding='unicode').strip()[:50]))
            else:
                labels[parent_name][actual_value] = dict(tag.attrib)
    for output_tag, tag_info in outputs.items():
        tag_info['inputs'] = []
        for input_tag_name in tag_info['to_name']:
            if input_tag_name not in inputs:
                logger.error(
                    f'to_name={input_tag_name} is specified for output tag name={output_tag}, '
                    'but we can\'t find it among input tags')
                continue
            tag_info['inputs'].append(inputs[input_tag_name])
        tag_info['labels'] = list(labels[output_tag])
        tag_info['labels_attrs'] = labels[output_tag]
    return outputs



OUTPUT_DIR = "/home/rchuzh/programming/image_labelling_shrdc/src/test/extra/test"
EXPORT_DIR = Path(
    "/home/rchuzh/programming/image_labelling_shrdc/src/test/extra/test").resolve()
JSON_DIR = "/home/rchuzh/Downloads/project-10-at-2021-07-27-07-49-62d7c421.json"

editor_config = """<View>
  <View style="display:flex;align-items:start;gap:8px;flex-direction:column-reverse">
    <Image name="img" value="$image" zoom="true" zoomControl="true" rotateControl="false" grid="true" brightnessControl="true" contrastControl="true"/>
    <View>
      <Filter toName="tag" minlength="0" name="filter"/>
      <RectangleLabels name="tag" toName="img" showInline="true">
        <Label value="Hello" background="#ff9ef4"/>
        <Label value="World" background="#8fd06d"/>
      </RectangleLabels>
    </View>
  </View>
</View>

"""
editor_config_IC="""<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <Choices name="choice" toName="image">
    <Choice value="QR"/>
    <Choice value="empty"/>
  </Choices>
</View>
"""
# editor_config=parse_config(editor_config)

converter = Converter(config=editor_config_IC, project_dir=None)
vars(converter)

task_json = """{
        "id": 2,
        "annotations": [
            {
                "id": 13,
                "completed_by": {
                    "id": 3,
                    "email": "ecyzc@nottingham.edu.my",
                    "first_name": "",
                    "last_name": ""
                },
                "result": [
                    {
                        "original_width": 3000,
                        "original_height": 4000,
                        "image_rotation": 0,
                        "value": {
                            "x": 17.600000000000005,
                            "y": 47.1,
                            "width": 32,
                            "height": 23.5,
                            "rotation": 0,
                            "rectanglelabels": [
                                "Hello"
                            ]
                        },
                        "id": "M4mae6iYcI",
                        "from_name": "tag",
                        "to_name": "img",
                        "type": "rectanglelabels"
                    },
                    {
                        "original_width": 3000,
                        "original_height": 4000,
                        "image_rotation": 0,
                        "value": {
                            "x": 55.86666666666667,
                            "y": 27.60000000000001,
                            "width": 15.333333333333334,
                            "height": 11.7,
                            "rotation": 0,
                            "rectanglelabels": [
                                "Hello"
                            ]
                        },
                        "id": "76y19AdsUB",
                        "from_name": "tag",
                        "to_name": "img",
                        "type": "rectanglelabels"
                    },
                    {
                        "original_width": 3000,
                        "original_height": 4000,
                        "image_rotation": 0,
                        "value": {
                            "x": 68,
                            "y": 42.3,
                            "width": 15.333333333333334,
                            "height": 12.5,
                            "rotation": 0,
                            "rectanglelabels": [
                                "Hello"
                            ]
                        },
                        "id": "Dh9hUuSc8Z",
                        "from_name": "tag",
                        "to_name": "img",
                        "type": "rectanglelabels"
                    }
                ],
                "was_cancelled": false,
                "ground_truth": false,
                "created_at": "2021-06-14T09:35:28.773001Z",
                "updated_at": "2021-06-18T07:52:24.910539Z",
                "lead_time": 17023.217,
                "prediction": {},
                "result_count": 0,
                "task": 2
            }
        ],
        "predictions": [],
        "file_upload": "IMG_20210315_181133.jpg",
        "data": {

            "image": "/home/rchuzh/Documents/aruco/train/IMG_20210315_181133.jpg"
        },
        "meta": {},
        "created_at": "2021-06-14T08:44:41.014786Z",
        "updated_at": "2021-06-18T07:52:24.858387Z",
        "project": 8
    }"""
tasks = json.loads(task_json)
with open(JSON_DIR, 'r') as f:
    json_dict = json.load(f)
# %%
iterator = converter.iter_from_json_file(json_dict, is_string=True)
# %%

for item in (iterator):
    item['output']

# %%
item

# %%
JSON_DIR.startswith('/home/')
# %%
converter.convert_to_json_min(json_dict, OUTPUT_DIR,
                         is_dir=False)
# %%
converter._get_supported_formats()

# %%
