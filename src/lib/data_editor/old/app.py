import streamlit as st
from streamlit_labelstudio import st_labelstudio

# st.set_page_config(layout='wide')
st.title("Label Studio")
config = """
      <View>
        <View style="padding: 25px; box-shadow: 2px 2px 8px #AAA;">
          <Image name="img" value="$image" width="100%" maxWidth="100%" brightnessControl="true" contrastControl="true" zoomControl="true" rotateControl="true"></Image>
        </View>
        <RectangleLabels name="tag" toName="img">
          <Label value="Comet"></Label>
          <Label value="Moon"></Label>
        </RectangleLabels>
      </View>
    """

interfaces = [
    "annotations:add-new",
    "annotations:delete",
    "annotations:menu",
    "controls",
    "panel",
    "predictions:menu",
    "side-column",
    "skip",
    "submit"
    "update",
],

user = {
    'pk': 1,
    'firstName': "Zhen Hao",
    'lastName': "Chu"
},

task = {
    'completions': [],
    'predictions': [],
    'id': 1,
    'data': {
        'image': "https://htx-misc.s3.amazonaws.com/opensource/label-studio/examples/images/nick-owuor-astro-nic-visuals-wDifg5xc9Z4-unsplash.jpg"
    }
}

import cv2
import base64


def ndarray_to_b64(ndarray):
    """
    converts a np ndarray to a b64 string readable by html-img tags 
    """
    img = cv2.cvtColor(ndarray, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


results_raw = st_labelstudio(config, interfaces, user, task, key='one')

if results_raw is not None:
    areas = [v for k, v in results_raw['areas'].items()]

    results = []
    for a in areas:
        results.append({'id': a['id'], 'x': a['x'], 'y': a['y'], 'width': a['width'],
                       'height': a['height'], 'label': a['results'][0]['value']['rectanglelabels'][0]})
    with st.beta_expander('Show Annotation Log'):

        st.table(results)
