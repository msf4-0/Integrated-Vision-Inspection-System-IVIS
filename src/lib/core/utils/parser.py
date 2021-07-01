"""
General Parser:
1. JSON
2. YAML
Date: 21/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)

"""
from pathlib import Path
from glob import glob, iglob

#-----------JSON PARSER-------------------#


#-----------YAML PARSER-------------------#

#-----------FILE HANDLER------------------#
def file_open(path):
    """general file opener"""
    with open(path, mode='r') as file:
        file = file.read()
    return file


def multi_file_open(path):
    """general multi-file opener"""

    file_list = []
    with open(path, mode='r') as files:
        file_list = files
    return file_list


def file_search(path=str(Path.home())):
    """
    File Search
    - Recursive true to search sub-directories with " ** "
    - Use wild-card for non-specific files ('*.extension')
    - CWD/*/*
    """
    file_list = []
    file_list = glob(pathname=path)
    # for file in glob(pathname=path):
    #     file_list.append(file)

    return file_list


def i_file_search(path=str(Path.home()), recursive=False):
    """Iterative File Search

    Args:
        path (str, optional): Path to file or folder. Defaults to str(Path.home()).

    Returns:
        List: List of files in that directory
    """

    file_list = []
    for file in iglob(pathname=path, recursive=recursive):
        file_list.append(file)
    return file_list
