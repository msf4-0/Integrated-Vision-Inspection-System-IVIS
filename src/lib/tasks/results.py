"""
To Obtain Annotations Results
Date: 21/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)

"""
import streamlit as st
from frontend.streamlit_labelstudio import st_labelstudio


results_raw = st_labelstudio(config, interfaces, user, task, key='Labelstudio')
st.write(results_raw)


if results_raw is not None:
    areas = [v for k, v in results_raw['areas'].items()]

    results = []
    for a in areas:
        results.append({'id': a['id'], 'x': a['x'], 'y': a['y'], 'width': a['width'],
                        'height': a['height'], 'label': a['results'][0]['value']['rectanglelabels'][0]})
    with st.beta_expander('Show Annotation Log'):

        st.table(results)
