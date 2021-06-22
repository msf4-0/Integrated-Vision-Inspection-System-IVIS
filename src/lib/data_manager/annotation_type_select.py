"""
Select Annotation Types
Date: 21/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
import streamlit as st
from pathlib import Path
import sys
import os

from annotation.annotation_template import loadAnnotationTemplate


def annotation_sel():
    """Select Annotation
    """
    # tuple of annotation types
    annotationType_list = ("", "Image Classification", "Object Detection with Bounding Boxes",
                           "Semantic Segmentation with Polygons", "Semantic Segmentation with Masks")
    annotationType_index = list(range(len(annotationType_list)))
    annotationType = st.selectbox("Template", annotationType_list, index=0, format_func=lambda x: 'Select an option' if x == '' else x,
                                  key="annotation_type", help="Please select the desired type of annotation")
    if annotationType is not "":
        annotationConfig_template = loadAnnotationTemplate(
            annotationType_list.index(annotationType) - 1)
        with st.beta_expander(label="template", expanded=False):
            st.write(annotationConfig_template)  # annotation config template
        return annotationType,annotationConfig_template
    else:
        return None
