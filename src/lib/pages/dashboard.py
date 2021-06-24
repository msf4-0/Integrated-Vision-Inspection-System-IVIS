"""
Title: Dashboard
Date: 23/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
import streamlit as st
from streamlit import cli as stcli
from pathlib import Path
from time import sleep
import sys
import pandas as pd
from core.utils.parser import file_search
from glob import glob


def write():
    st.write("# Dashboard")
    files = []

    # files = file_search(str(Path(PARENT, "src", "lib","*")))
    # st.write(files)
    # file_list = []
    # for file in files:
    #     file_list.append(str(Path(file).relative_to(Path(file).parents[1])))
    # st.write(file_list)


def main():
    write()


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
