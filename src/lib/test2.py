"""
    TEST
    """
from pathlib import Path
import streamlit as st
from streamlit import session_state as session_state

st.write(Path.cwd())

html_string = """
<svg height="210" width="500">
  <polygon points="200,10 250,190 160,210" style="fill:lime;stroke:purple;stroke-width:1" />
</svg>
"""

html_string2 = """
<div>
<div>
<svg width="16" height="16">
  <rect x="1" y="1" width="14" height="14" rx="1" color="#2876D4" style="stroke:pink;stroke-width:1;fill-opacity:0;stroke-opacity:0.9" />
  Sorry, your browser does not support inline SVG.  
</svg></div>
"aruco"
</div>"""
x=[1,2,3,4]
st.table(x)
st.markdown(html_string2, unsafe_allow_html=True)
