"""
Title: Annotation Template Handler
Date: 21/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
import sys
from os import path
from pathlib import Path
from yaml import full_load, load
import streamlit as st
#---------------Logger--------------#
import logging
FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
DATEFMT = '%d-%b-%y %H:%M:%S'
# logging.basicConfig(filename='test.log',filemode='w',format=FORMAT, level=logging.DEBUG)
logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                    stream=sys.stdout, datefmt=DATEFMT)
log = logging.getLogger()
#----------------------------------#

# annotation_file_dir = r'.\src\lib\annotation_template'
base_dir = Path(__file__).parent


# def loadTemplatedir(template_index):
#     """Load Annotation Template Directory

#     Args:
#         template_index ([type]): [description]

#     Returns:
#         [type]: [description]
#     """

#     folder = ['computer-vision']
#     file_path = Path(f'{folder[template_index]}')
#     log.info(file_path)
#     template_dir = Path(base_dir, file_path)
#     # relative_path=Path.joinpath("computer-vision")
#     # log.info(relative_path.is_dir())
#     # log.info(relative_path.parts)
#     # log.info(Path.cwd().joinpath("computer-vision"))
#     return template_dir

@st.cache
def loadAnnotationTemplate(template_index, template=None):
    """Load Annotation Template

    Args:
        template_index ([type]): [description]
        template ([type], optional): [description]. Defaults to None.
    """
    # log.info(f'Annotation Base Path: {base_dir}')

    template_list = ("image-classification",
                     "object-detection-bbox", "semantic-segmentation-polygon", "semantic-segmentation-mask")
    template = template_list[template_index]
    # file_dir = loadTemplatedir(template_index)
    file_path = Path(f'{template}')
    template_dir = Path(base_dir, file_path, Path('config.yml'))
    # log.info(template_dir)
    log.info(f"Loading template from {template_dir.relative_to(base_dir)}")
    template = loadTemplateConfig(template_dir)

    return template


def loadTemplateConfig(path):
    # if not path.exists(path):
    #     path = find_file(path)
    with open(path, mode='r', encoding='utf-8') as template:
        template = full_load(template)
    return template


# loadAnnotationTemplate()
# template = loadAnnotationTemplate(0)
# print(template)
