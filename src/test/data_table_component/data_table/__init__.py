import os
import streamlit as st
import streamlit.components.v1 as components
layout = 'wide'
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>>>>> TEMP for Logging >>>>>>>>
import logging
import sys
FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
DATEFMT = '%d-%b-%y %H:%M:%S'

# logging.basicConfig(filename='test.log',filemode='w',format=FORMAT, level=logging.INFO)
logging.basicConfig(format=FORMAT, level=logging.INFO,
                    stream=sys.stdout, datefmt=DATEFMT)

log = logging.getLogger()


_RELEASE = False


if not _RELEASE:

    _component_func = components.declare_component(
        "data_table", url="http://localhost:3001",)
else:

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component(
        "data_table", path=build_dir)


def data_table(data, key=None):
    component_value = _component_func(data=data, key=key, default=[])

    return component_value


if not _RELEASE:
    import streamlit as st
    import pandas as pd

    st.subheader("Dev Data Table with Material-UI")

    raw_data = {
        "First Name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
        "Last Name": ["Miller", "Jacobson", "Ali", "Milner", "Smith"],
        "Age": [42, 52, 36, 24, 73],
    }
    df = pd.DataFrame(raw_data, columns=["First Name", "Last Name", "Age"])

    rows = data_table(df)
    if rows:
        st.write("You have selected", rows)