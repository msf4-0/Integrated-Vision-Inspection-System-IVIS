
import streamlit as st
from streamlit import session_state as session_state





if 'flag' not in session_state:
    session_state.flag = 0

list1 = [1, 2, 3, 4, 5, 6]
list2 = [7, 8, 9, 10, 11, 12]
test_dict={'asfasdfa':list1,'asdfadadsf':list2}
test_list = [list1, list2]




st.write(test_list)

selection = st.selectbox(
    "test", options=test_list[session_state.flag], key='test')

st.write(selection)


def toggle():
    session_state.flag ^= 1
    return session_state.flag


st.button('toggle', key='toggle', on_click=toggle)
