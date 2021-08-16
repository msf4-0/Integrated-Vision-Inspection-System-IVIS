"""
    TEST
    """

from typing import NamedTuple
import streamlit as st
from streamlit import session_state as session_state
import streamlit.components.v1 as components
string1, string2 = ("New Project Info", "Editor Configuration")


class color_set(NamedTuple):
    border: str
    background: str


current_page = color_set('#0071BC', '#29B6F6')
non_current_page = color_set('#0071BC', None)
color = {'current_page': current_page, 'non_current_page': non_current_page}
def color_selection():
  selection = st.radio("Color Picker", options=color)
  st.write(type(color[selection]))
  html_string = f'''
    <style>
      .div1 {{
        display: flex;
        justify-content: space-evenly;

        padding: 10px;
        width: 30%;
        margin-left: auto;
        margin-right: auto;
      }}
      .div2 {{
      color:white;
        border-radius: 25px 0px 0px 25px;
        border-style: solid;
        border-color: {color[selection].border};
        background-color: {color[selection].background};
        padding: 10px;
        width: 100%;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
      }}
      .div3 {{
      color:white;
        border-radius: 0px 25px 25px 0px;
        border-style: solid;
        border-color: {color[selection].border};
        background-color: {color[selection]};
        padding: 10px;
        width: 100%;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
      }}
    </style>
    
      <div class="div1">
        <div class="div2">{string1}</div>
        <div class="div3">{string2}</div>
      </div>
    
  '''
  components.html(html_string)
