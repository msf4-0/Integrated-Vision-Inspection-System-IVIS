"""
Title: Dataset Compression
Date: 30/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""


from pathlib import Path
from os import path
import sys
# -------------

# SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
# sys.path.insert(0, str(Path(SRC, 'lib')))  # ./lib
# # print(sys.path[0])
# sys.path.insert(0, str(Path(Path(__file__).parent, 'module')))
# --------------

# import streamlit as st
# from streamlit import cli as stcli
from time import perf_counter

import logging
import psycopg2

#--------------------Logger-------------------------#
FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
DATEFMT = '%d-%b-%y %H:%M:%S'

# logging.basicConfig(filename='test.log',filemode='w',format=FORMAT, level=logging.DEBUG)
logging.basicConfig(format=FORMAT, level=logging.INFO,
                    stream=sys.stdout, datefmt=DATEFMT)

log = logging.getLogger()

#----------------------------------------------------#

