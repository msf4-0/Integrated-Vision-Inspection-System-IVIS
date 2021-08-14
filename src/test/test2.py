"""
    TEST
    """


import streamlit as st
from streamlit import session_state as session_state
from stqdm import stqdm
from time import sleep

x=[i for i in range(100)]

for i in stqdm(x):
    
    sleep(0.5)
