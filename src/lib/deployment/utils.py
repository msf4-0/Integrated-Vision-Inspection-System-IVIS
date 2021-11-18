from datetime import datetime
import os
import cv2
from imutils.video.webcamvideostream import WebcamVideoStream
from yaml import full_load

from paho.mqtt.client import Client
import streamlit as st
from streamlit import session_state

from core.utils.log import logger
from path_desc import MQTT_CONFIG_PATH

DATETIME_STR_FORMAT = "%Y-%m-%d_%H-%M-%S"

CONFIG_FILE_PATH = os.getenv("MQTT_CAMERA_CONFIG", MQTT_CONFIG_PATH)
with open(CONFIG_FILE_PATH) as f:
    CONFIG = full_load(f)

MQTT_BROKER = CONFIG["mqtt"]["broker"]
MQTT_PORT = CONFIG["mqtt"]["port"]
MQTT_QOS = CONFIG["mqtt"]["QOS"]

MQTT_TOPIC = CONFIG["camera"]["mqtt_topic"]
STOP_TOPIC = CONFIG["camera"]["stop_publish_topic"]
SAVE_TOPIC = CONFIG['save-captures']["save_frame_topic"]


def get_now_string(dt_format=DATETIME_STR_FORMAT) -> str:
    return datetime.now().strftime(dt_format)


def reset_camera():
    if 'camera' in session_state:
        if isinstance(session_state.camera, WebcamVideoStream):
            session_state.camera.stop()
            session_state.camera.stream.release()
        elif isinstance(session_state.camera, cv2.VideoCapture):
            session_state.camera.release()
        del session_state['camera']
    if 'working_ports' in session_state:
        del session_state['working_ports']


def on_connect(client, userdata, flags, rc):
    # The callback for when the client receives a CONNACK response from the server.
    logger.info(f"Connected to broker: {MQTT_BROKER}")
    logger.debug(f"Connected with result code {str(rc)} to MQTT broker "
                 f"on {MQTT_BROKER}")


def on_message(client, userdata, msg):
    """The callback for when a PUBLISH message is received from the server."""
    # if msg.topic != MQTT_TOPIC:
    #     return
    topic = msg.topic
    received = msg.payload.decode()
    result = f"Topic: {topic}; Received payload: '{received}'"
    # st.write(type(msg))
    # st.write(f"{msg.state = }")
    # st.write(dir(msg))  # info, payload, properties
    # st.write(result)
    logger.debug(f"{result = }")


def on_publish(client, userdata, mid):
    """Callback when a message is published"""
    # logger.debug(f"Publishing message, with mid={mid}")
    pass


def get_mqtt_client(client_id=''):
    """Return the MQTT session_state.client object."""
    if client_id:
        clean_session = False
    else:
        clean_session = True
    client = Client(client_id, clean_session)
    # client.connected_flag = False  # set flag
    client.on_connect = on_connect
    client.on_publish = on_publish
    return client


def disconnect_client():
    session_state.client.loop_stop()
    session_state.client.disconnect()
