"""
Title: Login Page
Date: 23/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
import streamlit as st
from streamlit import cli as stcli
from pathlib import Path
from time import sleep
import sys
# with __name__ is not "__main__":
#     PARENT = Path(__file__).parents[3]  # Parent folder
#     RESOURCE = Path(PARENT, "resources")

# left, mid, right = st.beta_columns([1,3,1])


def is_authenticated(username, password):  # WIP
    """Some function here to obtain User Credentials from Database
    WIP"""

    return username == "admin" and password == "shrdc"


def write():
    left, mid_login, right = st.beta_columns([1, 3, 1])
    with mid_login:
        with st.form("login", clear_on_submit=True):
            st.write("## Login")
            username = st.text_input(
                "Username", value="Please enter your username", key="username")
            pswrd = st.text_input("Password", value="",
                                  key="safe", type="password")
            submit_login = st.form_submit_button("Log In")
        # st.write(f"{username},{pswrd}")
        success_place = st.empty()
        if submit_login:
            if is_authenticated(username, pswrd):
                # st.balloons()
                # st.header("Welcome In")
                success_place.success("Welcome in")
                success_side = st.sidebar.empty()

                success_side.write("# Welcome In 👍")
                sleep(1)
                success_place.empty()
                success_side.empty()

            elif pswrd:
                st.error(
                    "User entered wrong username or password. Please enter again.")


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
