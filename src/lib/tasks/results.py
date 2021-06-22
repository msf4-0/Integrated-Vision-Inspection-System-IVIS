"""
To Obtain Annotations Results
Date: 21/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)

"""
import streamlit as st
from frontend.streamlit_labelstudio import st_labelstudio

interfaces = [
    "panel",
    "update",
    "submit",
    "controls",
    "side-column",
    "annotations:menu",
    "annotations:add-new",
    "annotations:delete",
    "predictions:menu",
],

#----------Object Detection with Bounding Boxes----------------#


def BBOX(config, user, task, interfaces=interfaces, key=None):
    """Obtain annotation results for Object Detection with Bounding Boxes 

    Args:
        config ([type]): [description]
        user ([type]): [description]
        task ([type]): [description]
        interfaces ([type], optional): [description]. Defaults to interfaces.
        key ([type], optional): [description]. Defaults to None.
    """
    results_raw = st_labelstudio(config, interfaces, user, task, key)
    st.write(results_raw)

    if results_raw is not None:
        areas = [v for k, v in results_raw['areas'].items()]

        results = []
        for a in areas:
            results.append({'id': a['id'], 'x': a['x'], 'y': a['y'], 'width': a['width'],
                            'height': a['height'], 'label': a['results'][0]['value']['rectanglelabels'][0]})
        with st.beta_expander('Show Annotation Log'):

            st.table(results)
            st.write(results_raw['areas'])
