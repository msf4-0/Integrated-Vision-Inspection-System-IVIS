import streamlit as st
from streamlit import session_state as session_state
from pathlib import Path
import sys
import pandas as pd
import numpy as np

SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
# TEST_MODULE_PATH = SRC / "test" / "test_page" / "module"

for path in sys.path:
    if str(LIB_PATH) not in sys.path:
        sys.path.insert(0, str(LIB_PATH))  # ./lib
    else:
        pass

from project.project_management import NewProject
from core.utils.code_generator import get_random_string
from core.utils.log import log_info
if "new_project" not in session_state:
    session_state.new_project=NewProject(get_random_string(8))

st.write(session_state.new_project.query_dataset_list())
st.write(session_state.new_project.dataset)
st.write(type(session_state.new_project.dataset[0][3]))
# df = pd.DataFrame(session_state.new_project.dataset, columns=[
#                 'ID', 'Name', 'Dataset Size', 'Date/Time'])
# st.write(df)
if session_state.new_project.dataset:
    df = pd.DataFrame(session_state.new_project.dataset, columns=[
                'ID', 'Name', 'Dataset Size', 'Date/Time'])
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    df.sort_values(by=['Date/Time'], inplace=True,
                        ascending=False, ignore_index=True)
    df.index.name = ('No.')
    styler = df.style.format(
    {
        "Date/Time": lambda t: t.strftime('%Y-%m-%d %H:%M:%S')
        
    }
)
    st.dataframe(styler)
   
df =session_state.new_project.create_dataset_dataframe()


st.dataframe(df)
