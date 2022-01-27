from csv import DictWriter
from dataclasses import dataclass
import os
from typing import Any, Callable, Dict, Iterator, List, Tuple
import cv2
from streamlit.uploaded_file_manager import UploadedFile
from yaml import full_load
from pathlib import Path

import streamlit as st
import numpy as np
from paho.mqtt.client import Client
from imutils.video.webcamvideostream import WebcamVideoStream
from streamlit import session_state

from core.utils.log import logger
from path_desc import MQTT_CONFIG_PATH


def image_from_buffer(buffer: bytes) -> np.ndarray:
    if not buffer:
        logger.error("Empty image buffer received")
        st.error("Please do not send empty image buffer. You can try to check "
                 "your input camera and try **Pause** and **Deploy** again.")
        st.stop()
    img_bytes = np.frombuffer(buffer, dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        logger.error("Error reading the input image buffer, please try to send the "
                     "image/frame in bytes.")
        st.error("Error reading the input image buffer, please try to send the "
                 "image/frame in bytes.")
        st.stop()
    return img


def read_images_from_uploaded(
        uploaded_imgs: List[UploadedFile]) -> Iterator[Tuple[np.ndarray, str]]:
    """Read from Streamlit List of UploadedFile and yield Tuple of image and filename."""
    for uploaded in uploaded_imgs:
        img = image_from_buffer(uploaded.getvalue())
        filename = uploaded.name
        yield img, filename


def image_to_bytes(frame: np.ndarray, channels: str = 'BGR') -> bytes:
    """Encode an image and convert into bytes"""
    if channels == 'RGB':
        # OpenCV needs BGR format
        out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        out = frame
    _, encoded_image = cv2.imencode('.png', out)
    bytes_array = encoded_image.tobytes()
    return bytes_array


def load_mqtt_config() -> Dict[str, str]:
    CONFIG_FILE_PATH = os.getenv("MQTT_CAMERA_CONFIG", MQTT_CONFIG_PATH)
    with open(CONFIG_FILE_PATH) as f:
        config = full_load(f)
    return config


@dataclass(eq=False)
class MQTTTopics:
    # publishing things to these topics
    # only `publish_frame` has different topic for each camera
    publish_frame: List[str]
    publish_results: str

    # subscribing to these topics to wait for input
    recv_frame: str
    start_publish: str
    stop_publish: str
    start_publish_frame: str
    stop_publish_frame: str
    save_frame: str
    start_record: str
    stop_record: str

    # NOTE: currently this is only used for dobot_arm_demo !
    dobot_view: str


CONFIG = load_mqtt_config()
ORI_PUBLISH_FRAME_TOPIC = CONFIG["main"]["publish_frame"]


@dataclass(eq=False)
class MQTTConfig:
    # 'mosquitto' is the service name defined in docker-compose.yml
    broker: str = 'mosquitto' if os.environ.get(
        "DOCKERCOMPOSE") else CONFIG["mqtt"]["broker"]
    # port 8883 instead of 1883 is used to avoid potential issues with local mosquitto broker,
    # this port is defined in mosquitto.conf
    port: int = 8883 if os.environ.get(
        "DOCKERCOMPOSE") else int(CONFIG["mqtt"]["port"])
    qos: int = int(CONFIG["mqtt"]["QOS"])
    topics: MQTTTopics = MQTTTopics(
        recv_frame=CONFIG["main"]["recv_frame"],
        # only this is a List[str], others just str
        publish_frame=[f'{ORI_PUBLISH_FRAME_TOPIC}_0'],
        publish_results=CONFIG["main"]["publish_results"],
        start_publish=CONFIG["camera"]["start_publish_topic"],
        stop_publish=CONFIG["camera"]["stop_publish_topic"],
        start_publish_frame=CONFIG["camera"]["start_publish_frame_topic"],
        stop_publish_frame=CONFIG["camera"]["stop_publish_frame_topic"],
        save_frame=CONFIG['save-captures']["save_frame_topic"],
        start_record=CONFIG['save-captures']['start_record_topic'],
        stop_record=CONFIG['save-captures']['stop_record_topic'],
        dobot_view='dobot/view'
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


def get_mqtt_client(client_id: str = '', clean_session: bool = True):
    """Return the MQTT client object."""
    client = Client(client_id, clean_session=clean_session)
    # client.connected_flag = False  # set flag
    client.on_connect = on_connect
    client.on_publish = on_publish
    return client


def create_csv_file_and_writer(
        csv_path: Path, results: List[Dict[str, Any]]):
    """Create session_state.csv_file (in open state and append mode) and 
    also session_state.csv_writer"""
    csv_dir = csv_path.parent
    if not csv_dir.exists():
        os.makedirs(csv_dir)

    if not csv_path.exists():
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
    for k in filter(lambda x: x.startswith('camera'), session_state.keys()):
        cap = session_state[k]
        if isinstance(cap, WebcamVideoStream):
            cap.stop()
            if 'ip' not in k:
                # IP camera seems like have issues if try to release like this
                cap.stream.release()
        elif isinstance(cap, cv2.VideoCapture):
            # cv2.VideoCapture instance
            cap.release()
        del session_state[k]


def reset_camera_and_ports():
    reset_camera()
    if 'working_ports' in session_state:
        del session_state['working_ports']


def reset_record_and_vid_writer():
    for k in filter(lambda x: x.startswith('vid_writer'), session_state.keys()):
        if isinstance(session_state[k], cv2.VideoWriter):
            # must release to properly close the video file
            session_state[k].release()
        del session_state[k]
    if 'record' in session_state:
        del session_state['record']


def reset_video_deployment():
    """Gracefully reset the session_state for `camera`, `deployed`, `mqtt_recv_frame`,
    `record`, `vid_writer`, `csv_file`, `csv_writer` but NOT `working_ports`"""
    logger.info("Resetting video deployment")

    reset_camera()
    # also pause the deployment when camera is reset
    if 'deployed' in session_state:
        del session_state['deployed']
    if 'mqtt_recv_frame' in session_state:
        del session_state['mqtt_recv_frame']
    reset_record_and_vid_writer()
    reset_csv_file_and_writer()


def reset_client():
    if 'client' in session_state:
        try:
            session_state.client.loop_stop()
            session_state.client.disconnect()
        except Exception as e:
            logger.error(f"Could not stop and disconnect client: {e}")
        del session_state['client']
        del session_state['client_connected']
        del session_state['added_video_cbs']
