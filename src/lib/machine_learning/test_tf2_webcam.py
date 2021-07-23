import numpy as np
import argparse
import os
import sys
from pathlib import Path
from queue import Queue,Empty
from typing import NamedTuple, List, Dict, Union
import cv2
import av
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import streamlit as st
from streamlit_webrtc import (
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

os.chdir(str(PROJECT_ROOT))
print(PROJECT_ROOT)


from core.utils.log import log_info, log_error  # logger
from module.test_detector import load_model

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
MODELS_DIR = '/home/rchuzh/programming/image_labelling_shrdc/resources/dl_models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model'
LABELMAP = '/home/rchuzh/programming/image_labelling_shrdc/resources/dl_models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/labelmap.pbtxt'
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


def main():
    class Detection(NamedTuple):
        name: str
        prob: float

    class Detector(VideoProcessorBase):
        confidence_threshold: float
        result_queue: "Queue[List[Detection]]"

        def __init__(self) -> None:
            self._model = load_model(MODELS_DIR)
            self.category_index = label_map_util.create_category_index_from_labelmap(LABELMAP,
                                                                                     use_display_name=True)
            self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD

        def _annotate_image(self, image, detections):
            log_info("**********************Visualising**********************")

            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                detections['detection_boxes'],
                detections['detection_classes'],
                detections['detection_scores'],
                self.category_index,
                instance_masks=detections.get(
                    'detection_masks_reframed', None),
                use_normalized_coordinates=True,
                min_score_thresh=self.confidence_threshold,
                line_thickness=8)
            return image

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format='bgr24')
            image = np.asarray(image)
            input_tensor = tf.convert_to_tensor(image)
            input_tensor = input_tensor[tf.newaxis, ...]

            detections = self._model(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(
                np.int64)

            if 'detection_masks' in detections:
                # Reframe the the bbox mask to the image size.
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detections['detection_masks'], detections['detection_boxes'],
                    image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    detection_masks_reframed > 0.5, tf.uint8)
                detections['detection_masks_reframed'] = detection_masks_reframed.numpy()
            annotated_image = self._annotate_image(image, detections)
            return av.VideoFrame.from_ndarray(annotated_image, format='bgr24')

    webrtc_ctx = webrtc_streamer(
        key="detector",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=Detector,
        async_processing=True,
    )

    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    )
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.confidence_threshold = confidence_threshold

    if st.checkbox("Show the detected labels", value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            # NOTE: The video transformation with object detection and
            # this loop displaying the result labels are running
            # in different threads asynchronously.
            # Then the rendered video frames and the labels displayed here
            # are not strictly synchronized.
            while True:
                if webrtc_ctx.video_processor:
                    try:
                        result = webrtc_ctx.video_processor.result_queue.get(
                            timeout=1.0
                        )
                    except Empty:
                        result = None
                    labels_placeholder.table(result)
                else:
                    break


main()
