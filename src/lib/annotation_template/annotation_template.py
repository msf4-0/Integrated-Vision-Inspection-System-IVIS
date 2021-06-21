"""
    Title: Annotation Template Handler
    Date: 21/6/2021
    Author: Chu Zhen Hao
    Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
import sys
from os import path
from pathlib import Path, PurePath
#---------------Logger--------------#
import logging
FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
DATEFMT = '%d-%b-%y %H:%M:%S'
# logging.basicConfig(filename='test.log',filemode='w',format=FORMAT, level=logging.DEBUG)
logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                    stream=sys.stdout, datefmt=DATEFMT)
log = logging.getLogger()
#----------------------------------#

annotation_file_dir = '.\src\lib\annotation_template'


def loadTemplatedir(template_index):
    base_dir = annotation_file_dir
    folder = ['computer-vision']
    file_path = f'\{folder[template_index]}'
    folder = PurePath(base_dir, file_path)
    # relative_path=Path.joinpath("computer-vision")
    # log.info(relative_path.is_dir())
    # log.info(relative_path.parts)
    # log.info(Path.cwd().joinpath("computer-vision"))
    folder


def loadAnnotationTemplate(template_index, template):
    file_dir = loadTemplatedir(template_index)
    log.info(f'Annotation Base Path: {annotation_file_dir}')

    annotation_template_dir = Path(annotation_file_dir, file_dir)
    # print(Path.cwd())


# loadAnnotationTemplate()
path = loadTemplatedir(0)
path
