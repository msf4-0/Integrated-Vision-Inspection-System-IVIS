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


def write():
    st.write("# Dashboard")


def main():
    write()


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        PARENT = Path(__file__).parents[3]  # Parent folder
        RESOURCE = Path(PARENT, "resources")
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
