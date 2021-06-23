"""
Title: Integrated Vision Inspection System
AUthor: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
# ----------Add sys path for modules----------------#
#
import sys
import os.path as osp
from pathlib import Path
ROOT = Path(__file__)  # ROOT folder
sys.path.insert(0, str(Path(ROOT, 'lib')))  # ./lib
print(sys.path[0])
#
#--------------------Logger-------------------------#
import logging
FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
DATEFMT = '%d-%b-%y %H:%M:%S'
# logging.basicConfig(filename='test.log',filemode='w',format=FORMAT, level=logging.DEBUG)
logging.basicConfig(format=FORMAT, level=logging.INFO,
                    stream=sys.stdout, datefmt=DATEFMT)
log = logging.getLogger()
#----------------------------------------------------#

import streamlit as st

# DEFINE Web APP page configuration
try:
    st.set_page_config(page_title="Label Studio Test",
                       page_icon="random", layout='wide')
except:
    st.beta_set_page_config(page_title="Label Studio Test",
                            page_icon="random", layout='wide')

"""
PAGES Dictionary
Import as modules from "./lib/pages"
"""
PAGES = {
    "LOGIN": "",
    "PROJECT": "",
    "DATASET": "",
    "INFERENCE": "",
}


def main():
    print("Hi")


if __name__ == "__main__":
    sys.exit(main())
