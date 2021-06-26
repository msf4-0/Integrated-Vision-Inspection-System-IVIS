"""
Name: Image Upload
Date: 8/6/2021 
"""
# --------------------------
# Add sys path for modules
import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), 'lib'))  # ./lib
# --------------------------

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import logging
from base64 import b64encode, decode
import io

#---------------Logger--------------#
import logging
FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
DATEFMT = '%d-%b-%y %H:%M:%S'
# logging.basicConfig(filename='test.log',filemode='w',format=FORMAT, level=logging.DEBUG)
logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                    stream=sys.stdout, datefmt=DATEFMT)
log = logging.getLogger()
#----------------------------------#
from threading import Thread

# -----------------------
import http.server
import socketserver
import socket

# hostname = socket.gethostname()
# IP = socket.gethostbyname(hostname)

# HOST, PORT = "localhost", 8000
# Handler = http.server.SimpleHTTPRequestHandler

# def client(HOST, PORT,Handler):
#     with socketserver.TCPServer((HOST, PORT), Handler) as server:
#         print("Serving at PORT ", PORT)
#         print("Your computer name is: ", hostname)
#         print("Your computer IP Address is: ", IP)
#         print("Local URL: " + f"http://localhost:{PORT}")
#         server.serve_forever()
# log.info(Handler)

# web=Thread(target=client, args=((HOST,PORT),Handler,))
# web.start()
# web.join()

# -------------------------------------

# from streamlit.util import index_
st.set_page_config(layout='wide')
#-----------Function Declaration---------------#


#----------------------------------------------#


st.title("Image Upload Test")

#-----------------Label Studio --------------------------------------#


# streamlit.file_uploader(label, type=None, accept_multiple_files=False, key=None, help=None)

# with st.sidebar.beta_container():
#     st.markdown("""
# # Individual Image Upload """)
#     uploaded_files = st.file_uploader(
#         label="Upload Image", type=['jpg', "png", "jpeg"], key=1)

# st.markdown("""
# ## Individual Image Upload """)
# if uploaded_files is not None:
#     # image = Image.open(uploaded_files)
#     if st.checkbox('Show image'):
#         st.subheader(f'Filename: {uploaded_files.name}')
#         st.image(uploaded_files, caption=uploaded_files.name)

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
if uploaded_files_multi is not None:
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
    st.image(uploaded_files_multi[image_name[image_sel]])
    st.write(uploaded_files_multi[image_name[image_sel]])
    # for image in uploaded_files_multi:
    #     st.write(image)
    #     st.subheader(f"Filename: {image.name}")
    # st.image(uploaded_files_multi[0])
    # st.image(uploaded_files_multi[1])
    # st.image(image)

    bb = uploaded_files_multi[image_name[image_sel]].read()
    b64code = b64encode(bb).decode('utf-8')
    data_url = f'data:image/jpg;base64,{b64code}'
    st.write(f"\"{data_url}\"")
else:
    pass
