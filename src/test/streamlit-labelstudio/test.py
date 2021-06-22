# --------------------------
# Add sys path for modules
import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), 'lib'))  # ./lib
# --------------------------

import streamlit as st
from streamlit_labelstudio import st_labelstudio

import numpy as np
import pandas as pd

from PIL import Image
import logging
from base64 import b64encode, decode
import io

from tasks.results import BBOX

#---------------Logger--------------#
import logging
FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
DATEFMT = '%d-%b-%y %H:%M:%S'
# logging.basicConfig(filename='test.log',filemode='w',format=FORMAT, level=logging.DEBUG)
logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                    stream=sys.stdout, datefmt=DATEFMT)
log = logging.getLogger()
#----------------------------------#


@st.cache
def dataURL_encoder(image):
    bb = image.read()
    b64code = b64encode(bb).decode('utf-8')
    data_url = f'data:image/jpg;base64,{b64code}'
    # st.write(f"\"{data_url}\"")
    return data_url or "None"


# --------------------------------

st.set_page_config(page_title="Label Studio Test",
                   page_icon="random", layout='wide')
st.markdown("""
# SHRDC Image Labelling Web APP 🎨
""")

with st.sidebar.beta_container():
    st.markdown("""
    ## Batch Image Upload """)

    # streamlit.file_uploader(label, type=None, accept_multiple_files=False, key=None, help=None)
    uploaded_files_multi = st.file_uploader(
        label="Upload Image", type=['jpg', "png", "jpeg"], accept_multiple_files=True, key=2)
    # if uploaded_files_multi is not None:
    # image_multi = Image.open(uploaded_files)
    # st.image(image_multi, caption="Uploaded Image")

st.markdown("""
    ## Batch Image Upload """)
if uploaded_files_multi:
    image_name = {}
    image_list = []
    i = 0
    for image in uploaded_files_multi:
        image_name[image.name] = i
        image_list.append(image.name)
        i += 1
    image_sel = st.sidebar.selectbox(
        "Select image", image_list)
    # with st.beta_expander('Show image'):
    st.write(uploaded_files_multi)
    st.subheader(f'Filename: {image_sel}')
    st.write(image_sel)
    # st.image(uploaded_files_multi[image_name[image_sel]])
    st.write(uploaded_files_multi[image_name[image_sel]])
    # for image in uploaded_files_multi:
    #     st.write(image)
    #     st.subheader(f"Filename: {image.name}")
    # st.image(uploaded_files_multi[0])
    # st.image(uploaded_files_multi[1])
    # st.image(image)
    data_url = dataURL_encoder(uploaded_files_multi[image_name[image_sel]])


else:
    data_url = "https://htx-misc.s3.amazonaws.com/opensource/label-studio/examples/images/nick-owuor-astro-nic-visuals-wDifg5xc9Z4-unsplash.jpg"

pass


config = """
      
  <View>
<Header value="Select label and start to click on image"/>
  <View style="display:flex;align-items:start;gap:8px;flex-direction:column-reverse">
    <Image name="img" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
    <View>
      <Filter toName="tag" minlength="0" name="filter"/>
      <RectangleLabels name="tag" toName="img" showInline="true">
        <Label value="Hello"/>
        <Label value="World"/>
      </RectangleLabels>
    </View>
  </View>
</View>
    """

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

user = {
    'pk': 1,
    'firstName': "Zhen Hao",
    'lastName': "Chu"
},

task = {
    'annotations': [],
    'predictions': [],
    'id': 1,
    'data': {
        # 'image': "https://htx-misc.s3.amazonaws.com/opensource/label-studio/examples/images/nick-owuor-astro-nic-visuals-wDifg5xc9Z4-unsplash.jpg"
        'image': f'{data_url}'
    }
}

# import cv2
# import base64


# def ndarray_to_b64(ndarray):
#     """
#     converts a np ndarray to a b64 string readable by html-img tags
#     """
#     img = cv2.cvtColor(ndarray, cv2.COLOR_RGB2BGR)
#     _, buffer = cv2.imencode('.png', img)
#     return base64.b64encode(buffer).decode('utf-8')

# log.info("load into component")

results_raw = st_labelstudio(config, interfaces, user, task, key='Labelstudio')
st.write(results_raw)
result_type = type(results_raw)
st.markdown(
    """
Results Type: {}\n
Area Type: {}\n

""".format(result_type, type(results_raw['areas'])))
st.write(results_raw['areas'])

if results_raw is not None:
    areas = [v for k, v in results_raw['areas'].items()]

    results = []
    for a in areas:
        results.append({'id': a['id'], 'x': a['x'], 'y': a['y'], 'width': a['width'],
                        'height': a['height'], 'label': a['results'][0]['value']['rectanglelabels'][0]})
    with st.beta_expander('Show Annotation Log'):

        st.table(results)
