"""
Title: Deployment
Date: 28/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)

Copyright (C) 2021 Selangor Human Resource Development Centre

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. 

Copyright (C) 2021 Selangor Human Resource Development Centre
SPDX-License-Identifier: Apache-2.0
========================================================================================

"""

from functools import partial
import json
import os
import shutil
import sys
from pathlib import Path
from time import perf_counter, sleep

import cv2
from imutils.video.webcamvideostream import WebcamVideoStream
from matplotlib import pyplot as plt
import numpy as np
from imutils.video.videostream import VideoStream
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state
from streamlit.report_thread import add_report_ctx
from yaml import full_load
# DEFINE Web APP page configuration
# layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

from path_desc import TEMP_DIR, chdir_root, MEDIA_ROOT, get_temp_dir
from core.utils.log import logger
from data_manager.database_manager import init_connection, db_fetchone
from machine_learning.utils import get_test_images_labels, get_tfod_test_set_data
from machine_learning.visuals import create_class_colors, create_color_legend
from deployment.deployment_management import DeploymentPagination, DeploymentType, Deployment
from deployment.utils import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, SAVE_TOPIC, STOP_TOPIC, get_mqtt_client, get_now_string, reset_camera
from core.utils.helper import get_directory_name, list_available_cameras

# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])

# >>>> Variable Declaration >>>>
new_project = {}  # store
place = {}
# DEPLOYMENT_TYPE = ("", "Image Classification", "Object Detection with Bounding Boxes",
#                    "Semantic Segmentation with Polygons", "Semantic Segmentation with Masks")


def kpi_string(text: str):
    return f"<h1 style='text-align: center; color:red;'>{text}</h1>"


def index(RELEASE=True):
    if 'deployment' not in session_state:
        st.sidebar.warning("You have not deployed any model yet.")
        st.sidebar.warning(
            "Please go to the Model Selection page and deploy a model first.")
        st.stop()

    if 'camera' not in session_state:
        # REMEMBER TO DELETE THIS if exists when entering this page
        session_state.camera = None
    if 'working_ports' not in session_state:
        session_state.working_ports = None

    # training paths obtained from Training.training_path in previous page
    TRAINING_PATHS = session_state.deployment.training_path
    DEPLOYMENT_TYPE = session_state.deployment.deployment_type

    # def back_to_model():
    #     session_state.deployment_pagination = DeploymentPagination.Models
    # st.sidebar.button(
    #     "Back to Model Selection",
    #     key='btn_back_model_select_deploy', on_click=back_to_model)

    input_type = st.sidebar.radio(
        'Choose the Type of Input',
        ('Image', 'Video'), key='input_type')

    if DEPLOYMENT_TYPE == 'Image Classification':
        kwargs = {}
    elif DEPLOYMENT_TYPE == 'Semantic Segmentation with Polygons':
        class_colors = create_class_colors(
            session_state.deployment.class_names)
        ignore_background = st.checkbox(
            "Ignore background", value=True, key='ignore_background',
            help="Ignore background class for visualization purposes")
        legend = create_color_legend(
            class_colors, bgr2rgb=False, ignore_background=ignore_background)
        st.markdown("**Legend**")
        st.image(legend)
        # convert to array
        class_colors = np.array(list(class_colors.values()),
                                dtype=np.uint8)
        kwargs = dict(class_colors=class_colors,
                      ignore_background=ignore_background)
    else:
        col1, _ = st.columns(2)
        conf_threshold = col1.number_input(
            "Minimum confidence threshold:",
            min_value=0.1,
            max_value=0.99,
            value=0.6,
            step=0.01,
            format='%.2f',
            key='conf_threshold',
            help=("If a prediction's confidence score exceeds this threshold, "
                  "then it will be displayed, otherwise discarded."),
        )
        kwargs = {'conf_threshold': conf_threshold}
    inference_pipeline = session_state.deployment.get_inference_pipeline(
        disable_timer=True, draw_result=True, **kwargs)

    if input_type == 'Image':
        logger.info("Loading test set data")
        if DEPLOYMENT_TYPE == 'Image Classification':
            X_test, *_ = get_test_images_labels(
                TRAINING_PATHS['test_set_pkl_file'],
                DEPLOYMENT_TYPE
            )
        elif DEPLOYMENT_TYPE == 'Semantic Segmentation with Polygons':
            X_test, _ = get_test_images_labels(
                TRAINING_PATHS['test_set_pkl_file'],
                DEPLOYMENT_TYPE
            )
        else:
            test_data_dir = TRAINING_PATHS['images'] / 'test'
            X_test = get_tfod_test_set_data(test_data_dir, return_xml_df=False)
        sample_image_path = X_test[0]

        uploaded_img = st.sidebar.file_uploader(
            "Upload an image", type=['jpg', 'jpeg', 'png'],
            key='image_uploader_deploy')
        st.markdown("**Upload an image for sample inference**")
        if not uploaded_img:
            st.markdown("**Sample Image**")
            img = cv2.imread(sample_image_path)
            filename = os.path.basename(sample_image_path)
        else:
            buffer = np.frombuffer(uploaded_img.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            filename = uploaded_img.name
            st.markdown(f"Uploaded Image: **{filename}**")

        # run inference on the image
        result = inference_pipeline(img)
        if DEPLOYMENT_TYPE == 'Image Classification':
            pred_classname, y_proba = result
            caption = (f"{filename}; "
                       f"Predicted: {pred_classname}; "
                       f"Score: {y_proba * 100:.1f}")
            st.image(img, channels='BGR', caption=caption)
        elif DEPLOYMENT_TYPE == 'Semantic Segmentation with Polygons':
            drawn_mask_output, pred_mask = result
            # convert to RGB for visualizing with Matplotlib
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            st.subheader(filename)
            fig = plt.figure()
            plt.subplot(121)
            plt.title("Original Image")
            plt.imshow(img)
            plt.axis('off')

            plt.subplot(122)
            plt.title("Predicted")
            plt.imshow(drawn_mask_output)
            plt.axis('off')

            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("___")
        else:
            img_with_detections, detections = result
            st.image(img_with_detections,
                     caption=f'Prediction: {filename}')

    elif input_type == 'Video':
        if DEPLOYMENT_TYPE == 'Image Classification':
            st.warning("""Sorry **Image Classification** type does not support video 
                input""")
            st.stop()

        max_allowed_fps = st.sidebar.number_input(
            'Maximum frame rate', 0.5, 60.0, 24.0, 0.1,
            key='selected_max_allowed_fps', on_change=reset_camera,
            help="""This is the maximum allowed frame rate that the 
                videostream will run at""")
        selected_width = st.sidebar.slider(
            'Width of video', 320, 1200, 320, 10, key='selected_width',
            help="This is the width of video for visualization purpose."
        )

        use_cam = st.sidebar.checkbox('Use video camera', value=False,
                                      key='cbox_use_camera', on_change=reset_camera)
        if use_cam:
            # TODO: test using streamlit-webrtc

            st.sidebar.button(
                "Reset camera", key='btn_reset_cam',
                on_click=reset_camera,
                help="Reset camera if you want to re-check the camera ports,  \nor if there "
                "is any problem with loading up the camera.")

            with st.sidebar.container():
                if not session_state.working_ports:
                    with st.spinner("Checking available camera ports ..."):
                        _, working_ports = list_available_cameras()
                        session_state.working_ports = working_ports.copy()
                camera_port = st.radio(
                    "Select a camera port", options=session_state.working_ports,
                    key='selected_cam_port', on_change=reset_camera)

            # only show these if a camera is not selected yet, to avoid keep checking
            # available ports
            if not session_state.camera:
                if st.sidebar.button("Start the selected camera", key='btn_start_cam'):
                    with st.spinner("Loading up camera ..."):
                        # NOTE: VideoStream does not work with filepath
                        try:
                            session_state.camera = VideoStream(
                                src=camera_port, framerate=max_allowed_fps).start()
                        except Exception as e:
                            st.error(
                                f"Unable to read from camera port {camera_port}")
                            logger.error(
                                f"Unable to read from camera port {camera_port}")
                            st.stop()
                        else:
                            sleep(2)  # give the camera some time to sink in
                else:
                    st.stop()
        else:
            video_file = st.sidebar.file_uploader(
                "Or upload a video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'],
                key='video_file_uploader')
            if not video_file:
                st.stop()

            if TEMP_DIR.exists():
                shutil.rmtree(TEMP_DIR)
            os.makedirs(TEMP_DIR)
            video_path = str(TEMP_DIR / video_file.name)
            st.write(f"{video_path = }")
            with open(video_path, 'wb') as f:
                f.write(video_file.getvalue())

            try:
                session_state.camera = cv2.VideoCapture(video_path)
                assert session_state.camera.isOpened(), "Video is unreadable"
                st.text('Input Video')
                st.video(video_path)
            except Exception as e:
                st.error(f"Unable to read from the video file: "
                         f"'{video_file.name}'")
                logger.error(f"Unable to read from the video file: "
                             f"'{video_file.name}' with error: {e}")
                st.stop()
            else:
                sleep(2)  # give the camera some time to sink in

        if session_state.camera:
            if isinstance(session_state.camera, WebcamVideoStream):
                stream = session_state.camera.stream
            else:
                stream = session_state.camera
            width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_input = int(stream.get(cv2.CAP_PROP_FPS))
            logger.debug(f"{width = }, {height = }, {fps_input = }")

        # ************************ MQTT STUFF ************************
        def stop_publish():
            logger.info("Stopping publishing ...")
            session_state.client_connected = False
            session_state.publishing = False
            logger.info("Stopped")
            # st.success("Stopped publishing")
            # cannot use this within mqtt callback
            # st.experimental_rerun()

        if 'client' not in session_state:
            session_state.client = get_mqtt_client()
            session_state.client_connected = False
            session_state.publishing = False

        st.sidebar.markdown("___")
        st.sidebar.subheader("MQTT Options")
        selected_qos = st.sidebar.radio(
            'MQTT QoS', (0, 1, 2), 1, key='selected_qos')

        if not session_state.client_connected:
            with st.spinner("Connecting to MQTT broker ..."):
                try:
                    session_state.client.connect(MQTT_BROKER, port=MQTT_PORT)
                except Exception as e:
                    st.error("Error connecting to MQTT broker")
                    logger.error(
                        f"Error connecting to MQTT broker {MQTT_BROKER}: {e}")
                    # return
                    st.stop()
                sleep(2)  # Wait for connection setup to complete
                logger.info("MQTT client connected successfully to "
                            f"{MQTT_BROKER} on port {MQTT_PORT}")
                session_state.client_connected = True

        def save_frame(client, userdata, msg):
            now = get_now_string()
            filename = f'test_frame-{now}.png'
            logger.info(f'Payload received for topic "{msg.payload}", '
                        f'saving frame as: "{filename}"')
            # need this to access to the frame from within mqtt callback
            nonlocal frame
            cv2.imwrite(filename, frame)

        logger.debug(f"{MQTT_TOPIC = }")
        logger.debug(f"{STOP_TOPIC = }")
        # on_message will serve as fallback when none matched
        # or use this to be more precise on the subscription topic filter
        session_state.client.message_callback_add(STOP_TOPIC, stop_publish)
        session_state.client.message_callback_add(SAVE_TOPIC, save_frame)
        session_state.client.subscribe(STOP_TOPIC, qos=selected_qos)
        session_state.client.subscribe(SAVE_TOPIC, qos=selected_qos)
        session_state.client.loop_start()
        # need to add this to avoid Missing ReportContext error
        # https://github.com/streamlit/streamlit/issues/1326
        add_report_ctx(session_state.client._thread)

        # ******************** OUTPUT VIDEO STUFF ********************
        st.subheader("Output Video")
        output_video_place = st.empty()
        publish_place = st.sidebar.empty()
        fps_col, width_col = st.columns(2)

        if publish_place.checkbox('Start publishing', key='cbox_publish'):
            logger.debug('Started publishing')
            session_state.publishing = True
        else:
            session_state.publishing = False
            # session_state.camera.release()

        show_labels = st.checkbox("Show the detected labels", value=True)
        if show_labels:
            result_col = st.container()
            with result_col:
                st.markdown("**Detected Results**")
                result_place = st.markdown("Coming up")
        with fps_col:
            st.markdown("**Frame Rate**")
            fps_place = st.markdown("0")
        with width_col:
            st.markdown("**Camera Image Width**")
            width_place = st.markdown("0")
        st.markdown("___")

        fps = 0
        i = 0
        prev_time = 0
        publish_func = partial(session_state.client.publish,
                               MQTT_TOPIC, qos=selected_qos)
        # while session_state.camera.grabbed:
        while True:
            if use_cam:
                frame = session_state.camera.read()
            else:
                ret, frame = session_state.camera.read()
                if not ret:
                    break

            # run inference on the frame
            result = inference_pipeline(frame)

            if DEPLOYMENT_TYPE == 'Semantic Segmentation with Polygons':
                drawn_mask_output, pred_mask = result
                img = drawn_mask_output
                results = session_state.deployment.get_segmentation_results(
                    pred_mask)
            else:
                img_with_detections, detections = result
                img = img_with_detections
                results = session_state.deployment.get_detection_results(
                    detections)

            # count FPS
            curr_time = perf_counter()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            output_video_place.image(
                frame, channels='BGR', width=selected_width)

            fps_place.markdown(kpi_string(int(fps)), unsafe_allow_html=True)
            width_place.markdown(kpi_string(width), unsafe_allow_html=True)
            if show_labels:
                result_place.table(results)

            if session_state.publishing:
                # payload = 'HELLO from Python'
                # logger.debug(f"{i}. Trying to publish message: '{payload}'")
                # frame = np.random.rand(10, 10)
                payload = json.dumps(results)
                info = publish_func(payload=payload)
            if fps > max_allowed_fps:
                sleep(1 / (fps - max_allowed_fps))

        reset_camera()
        if TEMP_DIR.exists():
            logger.debug("Removing temporary directory")
            shutil.rmtree(TEMP_DIR)


def main():
    # False for debugging
    index(RELEASE=False)


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
