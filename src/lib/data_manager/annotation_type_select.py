"""
Select Annotation Types
Date: 21/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
import streamlit as st
from pathlib import Path
import xml.dom.minidom as minidom


from annotation.annotation_template import load_annotation_template


def annotation_sel():
    """Select Annotation
    """
    # tuple of annotation types
    annotationType_list = (" ", "Image Classification", "Object Detection with Bounding Boxes",
                           "Semantic Segmentation with Polygons")
    annotationType_index = list(range(len(annotationType_list)))
    annotationType = st.selectbox("Template", annotationType_list, index=0, format_func=lambda x: 'Select an option' if x == ' ' else x,
                                  key="annotation_type", help="Please select the desired type of annotation")
    if annotationType is not " ":
        annotationConfig_template = load_annotation_template(
            annotationType_list.index(annotationType) - 1)
        with st.expander(label="template", expanded=False):
            template = (minidom.parseString(
                annotationConfig_template['config']))
            # annotation config template
            st.code(template.toprettyxml())
        return annotationType, annotationConfig_template
    else:
        annotationType = annotationConfig_template = None
        return annotationType, annotationConfig_template
