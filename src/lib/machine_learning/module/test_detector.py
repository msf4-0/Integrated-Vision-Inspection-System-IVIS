import streamlit as st
from google.protobuf.pyext._message import RepeatedCompositeContainer
import tensorflow as tf

@st.cache
def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model
