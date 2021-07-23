"""
Title: Object Detection TF2
Date: 1/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
Description: Object detection
"""


import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.utils import label_map_util

import sys
import os
from pathlib import Path

from time import perf_counter
import numpy as np
import cv2
from typing import List, Dict, NamedTuple
from queue import Queue
import av
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state as session_state
from streamlit_webrtc import (
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
from tensorflow.python.client import session

# import urllib.request
# import tarfile

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

os.chdir(str(PROJECT_ROOT))
print(PROJECT_ROOT)
# >>>> User-defined Modules >>>>
# from frame_overlay import draw_overlay
from module.performance_metrics import PerformanceMetrics
from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
# from data_manager.database_manager import init_connection
from module.detector import Model,load_model
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Setup WebRTC >>>>
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)
# <<<< Setup WebRTC <<<<


'''
DATA_DIR = os.path.join(os.getcwd(), 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
for dir in [DATA_DIR, MODELS_DIR]:
    if not os.path.exists(dir):
        os.mkdir(dir)
'''


def main():
    # ADD YOUR DIRECTORY HERE!!!!!
    DATA_DIR = '/home/chuzh/Documents/TensorFlow/workspace/training_demo'
    MODELS_DIR = '/home/rchuzh/programming/image_labelling_shrdc/resources/dl_models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'

    PATH_TO_CKPT = os.path.join(MODELS_DIR, 'checkpoint/')
    PATH_TO_CFG = os.path.join(MODELS_DIR, 'pipeline.config')

    LABEL_FILENAME = 'labelmap.pbtxt'
    PATH_TO_LABELS = os.path.join(MODELS_DIR, LABEL_FILENAME)
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    class Detection(NamedTuple):
        name: str
        prob: float

    class Detector(VideoProcessorBase):
        confidence_threshold: float
        detection_results: "Queue[List[Detection]]"

        def __init__(self) -> None:
            '''
            detector_config=['models_dir':MODELS_DIR,'path_to_ckpt':PATH_TO_CKPT,'label_filename':LABEL_FILENAME,'path_to_labels':PATH_TO_LABELS]
            '''
            log_info("********************** INIT DETECTOR **********************")
            self.confidence_threshold = 0.5
            self.detection_model = load_model()
            self.category_index = {}
            self.detection_results = Queue()
            self.current_fps = 0

        @tf.function
        def detect_fn(self, image: np.ndarray):
            """Detect objects in image."""
            log_info(" ********************** Detecting ********************** ")
            image, shapes = self.detection_model.preprocess(image)
            prediction_dict = self.detection_model.predict(image, shapes)
            detections = self.detection_model.postprocess(prediction_dict, shapes)

            return detections

        def _annotate_frame(self, image: np.ndarray, detections, current_fps: float, label_id_offset=1):
            image_np_with_detections = image.copy()  # ndarray.copy()

            # TensorFlow visualiser
            # TODO get queue of results from TensorFlow viz
            log_info("**********************Visualising**********************")
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'][0].numpy(),
                (detections['detection_classes']
                [0].numpy() + label_id_offset).astype(int),
                detections['detection_scores'][0].numpy(),
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=self.confidence_threshold,
                agnostic_mode=False)
            log_info(" ********************** Overlay ********************** ")
            # image_np_with_detections = draw_overlay(
            #     image_np_with_detections, current_fps, ' ')
            return image_np_with_detections

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            log_info(">>>> Enter recv")
            image: np.ndarray = frame.to_ndarray(format="bgr24")  # get numpy array
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image, axis=0)

            # Input Tensor
            input_tensor = tf.convert_to_tensor(
                image_np_expanded, dtype=tf.float32)
            detections = self.detect_fn(input_tensor)

            image_np_with_detections = self._annotate_frame(
                image, detections, 0)
            # TODO get queue of results from TensorFlow viz
            log_info("<<<< Exit recv ")
            return av.VideoFrame.from_ndarray(image_np_with_detections, format='bgr24')

    #---------------------- Load the model -------------------------#

    load_start_time = perf_counter()  # program loading start timestamp

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging

    tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

    # Enable GPU dynamic memory allocation
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # # Load pipeline config and build a detection model
    # configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    # model_config = configs['model']
    # detection_model = model_builder.build(
    #     model_config=model_config, is_training=False)

    # # Restore checkpoint
    # ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    # ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

    # @tf.function
    # def detect_fn(image):
    #     """Detect objects in image."""

    #     image, shapes = detection_model.preprocess(image)
    #     prediction_dict = detection_model.predict(image, shapes)
    #     detections = detection_model.postprocess(prediction_dict, shapes)

    #     return detections, prediction_dict, tf.reshape(shapes, [-1])

    # #--------------- Load label map data (for plotting)---------------------#
    # category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
    #                                                                     use_display_name=True)

    if 'model' not in session_state:
        session_state.model = Model(MODELS_DIR)
    # session_state.model.load_config()
    # session_state.model.load_model
    # -----------------------Start Video Capture--------------------------#
    link = "http://192.168.1.105:4747/video"  # IP webcam
    local_link = "http://127.0.0.1:4747"    # not use -> /dev/video2

    # TODO : >>>>>>>>>>>>>>>>>>>>GET VIDEO FROM WEBRTC>>>>>>>>>>>>>>>>>>>>
    cap = cv2.VideoCapture(0)

    perfMetric = PerformanceMetrics()  # instantiate performance metrics class

    load_end_time = perf_counter()  # program loading end timestamp
    load_time = load_end_time - load_start_time
    # computes loading time of the program
    log_info(f"Program Loading time = {load_time}s")
    # while cap.isOpened():

    perfMetric.start_time = perf_counter()
    #     ret, image_np = cap.read()  # Read frame from camera
    #     image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # convert to RGB
    #     # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    #     image_np_expanded = np.expand_dims(image_np, axis=0)

    #     input_tensor = tf.convert_to_tensor(
    #         image_np_expanded, dtype=tf.float32)
    #     detections, predictions_dict, shapes = detect_fn(input_tensor)

    #     label_id_offset = 1
    #     image_np_with_detections = image_np.copy()  # frame
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=Detector,
        async_processing=True,
    )
    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    )
    while webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.confidence_threshold = confidence_threshold
        webrtc_ctx.video_processor.detection_model = session_state.model.detection_model
        webrtc_ctx.video_processor.category_index = session_state.model.category_index

        current_latency, current_fps = perfMetric.update(perfMetric.start_time)
        if (current_latency and current_fps) is not None:
            log_info(
                f"{'Current Latency=':<4} {(current_latency* 1e3):4.1f}ms {'|':^8} {'Current FPS=':<4} {current_fps:4.1f}")
            webrtc_ctx.video_processor.current_fps = current_fps
    # >>>>>>>>>> ADD ANNOTATIONS TO VIDEO FRAMES >>>>>>>>>>>#
    # viz_utils.visualize_boxes_and_labels_on_image_array(
    #     image_np_with_detections,
    #     detections['detection_boxes'][0].numpy(),
    #     (detections['detection_classes']
    #         [0].numpy() + label_id_offset).astype(int),
    #     detections['detection_scores'][0].numpy(),
    #     category_index,
    #     use_normalized_coordinates=True,
    #     max_boxes_to_draw=200,
    #     min_score_thresh=.60,
    #     agnostic_mode=False)
    # image_np_with_detections = draw_overlay(
    #     image_np_with_detections, current_fps, ' ')
    # Display output
    # cv2.imshow('object detection', cv2.resize(
    #     image_np_with_detections, (1280, 960)))
    # cv2.imshow('object detection', image_np_with_detections)

    # # QUIT PROGRAM
    # # *********************************************
    # # need to add a flag from Streamlit
    # key = cv2.waitKey(1)

    # ESC_KEY = 27
    # # Quit.
    # if key in {ord('q'), ord('Q'), ESC_KEY}:
    #     break
        # *********************************************
    # perfMetric.print_total()
    cap.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
