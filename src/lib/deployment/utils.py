from csv import DictWriter
from dataclasses import dataclass, field
from datetime import datetime
import os
from typing import Any, Dict, List, NamedTuple
import cv2
from yaml import full_load
from pathlib import Path

from paho.mqtt.client import Client
from imutils.video.webcamvideostream import WebcamVideoStream
import streamlit as st
from streamlit import session_state

from core.utils.log import logger
from path_desc import MQTT_CONFIG_PATH


def load_mqtt_config() -> Dict[str, str]:
    CONFIG_FILE_PATH = os.getenv("MQTT_CAMERA_CONFIG", MQTT_CONFIG_PATH)
    with open(CONFIG_FILE_PATH) as f:
        config = full_load(f)
    return config


class MQTTTopics(NamedTuple):
    publish_results: str
    start_publish: str
    stop_publish: str
    save_frame: str
    start_record: str
    stop_record: str


CONFIG = load_mqtt_config()


@dataclass(init=False, frozen=True, eq=False)
class MQTTConfig:
    # 'mosquitto' is the service name defined in docker-compose.yml
    broker: str = 'mosquitto' if os.environ.get(
        "DOCKERCOMPOSE") else CONFIG["mqtt"]["broker"]
    # port 8883 instead of 1883 is used to avoid potential issues with local mosquitto broker,
    # this port is defined in mosquitto.conf
    port: str = 8883 if os.environ.get(
        "DOCKERCOMPOSE") else CONFIG["mqtt"]["port"]
    qos: str = CONFIG["mqtt"]["QOS"]
    topics: MQTTTopics = MQTTTopics(
        publish_results=CONFIG["camera"]["publish_results"],
        start_publish=CONFIG["camera"]["start_publish_topic"],
        stop_publish=CONFIG["camera"]["stop_publish_topic"],
        save_frame=CONFIG['save-captures']["save_frame_topic"],
        start_record=CONFIG['save-captures']['start_record_topic'],
        stop_record=CONFIG['save-captures']['stop_record_topic']
    )


def on_connect(client, userdata, flags, rc):
    mqtt_conf = MQTTConfig()
    # The callback for when the client receives a CONNACK response from the server.
    logger.info(f"Connected to broker: {mqtt_conf.broker}")
    logger.debug(f"Connected with result code {str(rc)} to MQTT broker "
                 f"on {mqtt_conf.broker}")


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
    """Return the MQTT client object."""
    client = Client(client_id, clean_session=True)
    # client.connected_flag = False  # set flag
    client.on_connect = on_connect
    client.on_publish = on_publish
    return client


def create_csv_file_and_writer(
        csv_path: Path, results: List[Dict[str, Any]], new_file: bool = True):
    """Create session_state.csv_file (in open state and append mode) and 
    also session_state.csv_writer"""
    csv_dir = csv_path.parent
    if not csv_dir.exists():
        os.makedirs(csv_dir)
    if new_file:
        with open(csv_path, 'w') as csv_file:
            session_state.csv_writer = DictWriter(
                csv_file, fieldnames=results[0].keys())
            session_state.csv_writer.writeheader()
            for row in results:
                session_state.csv_writer.writerow(row)
    if session_state.csv_file and not session_state.csv_file.closed:
        session_state.csv_file.close()
    session_state.csv_file = open(csv_path, 'a')
    session_state.csv_writer = DictWriter(
        session_state.csv_file, fieldnames=results[0].keys())


def reset_csv_file_and_writer():
    if 'csv_file' in session_state:
        if session_state.csv_file and not session_state.csv_file.closed:
            session_state.csv_file.close()
        del session_state['csv_file']
    if 'csv_writer' in session_state:
        del session_state['csv_writer']


def reset_camera():
    if 'camera' in session_state:
        if isinstance(session_state.camera, WebcamVideoStream):
            session_state.camera.stop()
            session_state.camera.stream.release()
        elif isinstance(session_state.camera, cv2.VideoCapture):
            session_state.camera.release()
        del session_state['camera']


def reset_camera_ports():
    if 'working_ports' in session_state:
        del session_state['working_ports']


def reset_record_and_vid_writer():
    if 'vid_writer' in session_state:
        if isinstance(session_state.vid_writer, cv2.VideoWriter):
            session_state.vid_writer.release()
        del session_state['vid_writer']
    if 'record' in session_state:
        del session_state['record']


def reset_client():
    if 'client' in session_state:
        try:
            session_state.client.loop_stop()
            session_state.client.disconnect()
        except Exception as e:
            logger.error(f"Could not stop and disconnect client: {e}")
        del session_state['client']
        del session_state['client_connected']
