import os
import streamlit.components.v1 as components
import streamlit as st

_RELEASE = True


if not _RELEASE:
    _component_func = components.declare_component(

        "color_extract",

        url="http://localhost:3000",
    )
else:

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component(
        "color_extract", path=build_dir)


def color_extract(nothing=None, key=None):

    component_value = _component_func(nothing=nothing, key=key, default=None)

    return component_value


# if not _RELEASE:
#     import streamlit as st

#     st.subheader("Color extract")
#     hex_code = color_extract(key='dev')
#     st.write(hex_code)
