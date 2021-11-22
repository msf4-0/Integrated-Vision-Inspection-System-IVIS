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

from datetime import datetime
from functools import partial
import json
import os
import shutil
import sys
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Dict

import cv2
from imutils.video.webcamvideostream import WebcamVideoStream
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state
from streamlit.report_thread import add_report_ctx

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
from data_manager.dataset_management import Dataset
from deployment.deployment_management import DeploymentConfig, DeploymentPagination, DeploymentType, Deployment
from deployment.utils import MQTTConfig, create_csv_file_and_writer, get_mqtt_client, reset_camera, reset_camera_ports
from core.utils.helper import Timer, get_directory_name, get_now_string, get_today_string, list_available_cameras

# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])

# >>>> Variable Declaration >>>>
new_project = {}  # store
place = {}
# DEPLOYMENT_TYPE = ("", "Image Classification", "Object Detection with Bounding Boxes",
#                    "Semantic Segmentation with Polygons", "Semantic Segmentation with Masks")


def kpi_format(text: str):
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
    # TRAINING_PATHS = session_state.deployment.training_path
    DEPLOYMENT_TYPE = session_state.deployment.deployment_type

    # def back_to_model():
    #     session_state.deployment_pagination = DeploymentPagination.Models
    # st.sidebar.button(
    #     "Back to Model Selection",
    #     key='btn_back_model_select_deploy', on_click=back_to_model)

    # deploy_config_path = session_state.deployment.get_config_path()
    if 'deployment_conf' not in session_state:
        session_state.deployment_conf = DeploymentConfig()

    deploy_conf = session_state.deployment_conf

    options = ('Image', 'Video')
    index = options.index(deploy_conf.input_type)
    input_type = st.sidebar.radio(
        'Choose the Type of Input', options,
        index=index, key='input_type')
    deploy_conf.input_type = input_type

    options_col, _ = st.columns(2)

    if DEPLOYMENT_TYPE == 'Image Classification':
        pipeline_kwargs = {}
    elif DEPLOYMENT_TYPE == 'Semantic Segmentation with Polygons':
        class_colors = create_class_colors(
            session_state.deployment.class_names)
        ignore_background = deploy_conf.ignore_background
        ignore_background = st.checkbox(
            "Ignore background", value=False, key='ignore_background',
            help="Ignore background class for visualization purposes")
        deploy_conf.ignore_background = ignore_background
        legend = create_color_legend(
            class_colors, bgr2rgb=False, ignore_background=ignore_background)
        st.markdown("**Legend**")
        st.image(legend)
        # convert to array
        class_colors = np.array(list(class_colors.values()),
                                dtype=np.uint8)
        pipeline_kwargs = {'class_colors': class_colors,
                           'ignore_background': ignore_background}
    else:
        conf_threshold = options_col.slider(
            "Confidence threshold:",
            min_value=0.1,
            max_value=0.99,
            value=deploy_conf.confidence_threshold,
            step=0.01,
            format='%.2f',
            key='conf_threshold',
            help=("If a prediction's confidence score exceeds this threshold, "
                  "then it will be displayed, otherwise discarded."),
        )
        deploy_conf.confidence_threshold = conf_threshold
        pipeline_kwargs = {'conf_threshold': conf_threshold}

    if input_type == 'Image':
        image_type = st.sidebar.radio(
            "Select type of image",
            ("Image from project datasets", "Uploaded Image"),
            key='select_image_type')
        if image_type == "Image from project datasets":
            project_datasets = session_state.project.data_name_list.keys()
            selected_dataset = st.sidebar.selectbox(
                "Select a dataset from project datasets",
                project_datasets, key='selected_dataset')

            project_image_names = session_state.project.data_name_list[selected_dataset]
            total_images = len(project_image_names)
            default_n = 10 if total_images >= 10 else total_images
            n_images = st.sidebar.number_input(
                "Select number of images to load",
                1, total_images, default_n, 5, key='n_images',
                help=(f"Total images in the project is **{total_images}**.  \n"
                      "Choose a lower value to reduce memory consumption."))
            project_image_names = project_image_names[:n_images]

            filename = st.sidebar.selectbox(
                "Select a sample image from the project dataset",
                project_image_names, key='filename')
            image_idx = project_image_names.index(filename)
            filename = project_image_names[image_idx]
            dataset_path = Dataset.get_dataset_path(selected_dataset)
            image_path = str(dataset_path / filename)

            st.markdown("**Selected image from project dataset**")
            img = cv2.imread(image_path)
        else:
            uploaded_img = st.sidebar.file_uploader(
                "Upload an image", type=['jpg', 'jpeg', 'png'],
                key='image_uploader_deploy')
            if not uploaded_img:
                st.stop()

            buffer = np.frombuffer(uploaded_img.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            filename = uploaded_img.name
            st.markdown(f"Uploaded Image: **{filename}**")

        if DEPLOYMENT_TYPE != 'Semantic Segmentation with Polygons':
            ori_image_width = img.shape[1]
            image_width = st.sidebar.slider(
                "Select width of image to resize for display",
                35, 1000, 500, 5, key='selected_width')
            # help=f'Original image width is **{ori_image_width}**.')
            st.sidebar.markdown(f"Original image width: **{ori_image_width}**")

        inference_pipeline = session_state.deployment.get_inference_pipeline(
            draw_result=True, **pipeline_kwargs)

        # run inference on the image
        with st.spinner("Running detection ..."):
            with Timer("Inference on image"):
                result = inference_pipeline(img)
        if DEPLOYMENT_TYPE == 'Image Classification':
            pred_classname, y_proba = result
            caption = (f"{filename}; "
                       f"Predicted: {pred_classname}; "
                       f"Score: {y_proba * 100:.1f}")
            st.image(img, channels='BGR', width=image_width, caption=caption)
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
            st.image(img_with_detections, width=image_width,
                     caption=f'Detection result for: {filename}')

    elif input_type == 'Video':
        if DEPLOYMENT_TYPE == 'Image Classification':
            st.warning("""Sorry **Image Classification** type does not support video 
                input""")
            st.stop()

        # Does not seem to work properly
        # max_allowed_fps = st.sidebar.slider(
        #     'Maximum frame rate', 1, 60, 24, 1,
        #     key='selected_max_allowed_fps', on_change=reset_camera,
        #     help="""This is the maximum allowed frame rate that the
        #         videostream will run at.""")
        selected_width = st.sidebar.slider(
            'Width of video', 320, 1200, deploy_conf.video_width, 10,
            key='selected_width',
            help="This is the width of video for visualization purpose."
        )
        deploy_conf.video_width = int(selected_width)

        use_cam = st.sidebar.checkbox('Use video camera', value=deploy_conf.use_camera,
                                      key='cbox_use_camera', on_change=reset_camera)
        deploy_conf.use_camera = use_cam
        if use_cam:
            # TODO: test using streamlit-webrtc

            st.sidebar.button(
                "Stop and reset camera", key='btn_reset_cam',
                on_click=reset_camera,
                help=("Reset camera if there is any problem with loading up  \n"
                      "the camera. Note that this is extremely important  \n"
                      "to ensure your camera is properly stopped and the  \n"
                      "camera access is given back to your system."))

            with st.sidebar.container():
                options = ('USB Camera', 'IP Camera')
                index = options.index(deploy_conf.camera_type)
                camera_type = st.radio(
                    "Select type of camera", options, index=0,
                    key='camera_type', on_change=reset_camera)
                deploy_conf.camera_type = camera_type
                if camera_type == 'USB Camera':
                    if not session_state.working_ports:
                        with st.spinner("Checking available camera ports ..."):
                            _, working_ports = list_available_cameras()
                            session_state.working_ports = working_ports.copy()
                    st.button("Refresh camera ports", key='btn_refresh',
                              on_click=reset_camera_ports)
                    video_source = st.radio(
                        "Select a camera port",
                        options=session_state.working_ports,
                        index=deploy_conf.camera_port,
                        key='selected_cam_port', on_change=reset_camera)
                    deploy_conf.camera_port = int(video_source)
                else:
                    video_source = st.text_input(
                        "Enter the IP address", value=deploy_conf.ip_cam_address)
                    deploy_conf.ip_cam_address = video_source.strip()

            # only show these if a camera is not selected yet, to avoid keep checking
            # available ports
            if not session_state.camera:
                if st.sidebar.button(
                    "🛠️ Deploy model", key='btn_start_cam',
                        help='Deploy your model with the selected camera source'):
                    with st.spinner("Loading up camera ..."):
                        # NOTE: VideoStream does not work with filepath
                        try:
                            session_state.camera = WebcamVideoStream(
                                src=video_source).start()
                        except Exception as e:
                            st.error(
                                f"Unable to read from video source {video_source}")
                            logger.error(
                                f"Unable to read from video source {video_source}: {e}")
                            st.stop()
                        else:
                            sleep(2)  # give the camera some time to sink in
                        # rerun just to avoid displaying unnecessary buttons
                        st.experimental_rerun()
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
        else:
            st.stop()

        # **************************** MQTT STUFF ****************************
        if 'client' not in session_state:
            session_state.client = get_mqtt_client()
            session_state.client_connected = False

        if 'publishing' not in session_state:
            session_state.publishing = True
            session_state.csv_writer = None
            session_state.csv_file = None

            session_state.record = False
            session_state.vid_writer = None

            session_state.refresh = False

        def create_video_writer_if_not_exists():
            if not session_state.vid_writer:
                logger.info("Creating video file to record to")
                downloads_folder = Path.home() / "Downloads"
                # NOTE: THIS VIDEO SAVE FORMAT IS VERY PLATFORM DEPENDENT
                # TODO: THE VIDEO FILE MIGHT NOT SAVE PROPERLY
                # usually either MJPG + .avi, or XVID + .mp4
                FOURCC = cv2.VideoWriter_fourcc(*"XVID")
                filename = f"video_{get_now_string()}.mp4"
                video_save_path = str(downloads_folder / filename)
                # st.info(f"Video is being saved to **{video_save_path}**")
                logger.info(f"Video is being saved to '{video_save_path}'")
                # this FPS value is the FPS of the output video file,
                # note that if this value is much higher than the fps
                # during the inference time, the output video will look
                # like it's moving very very fast
                FPS = 24
                session_state.vid_writer = cv2.VideoWriter(
                    video_save_path, FOURCC, FPS,
                    (width, height), True)

        def start_publish_cb(client, userdata, msg):
            logger.info("Start publishing")
            session_state.publishing = True
            # use this to refresh the page once to show widget changes
            session_state.refresh = True

        def stop_publish_cb(client, userdata, msg):
            logger.info("Stopping publishing ...")
            # session_state.client_connected = False
            session_state.publishing = False
            session_state.deployment_conf.publishing = False
            logger.info("Stopped")
            session_state.refresh = True
            # st.success("Stopped publishing")
            # cannot use this within mqtt callback
            # st.experimental_rerun()

        def save_frame_cb(client, userdata, msg):
            now = get_now_string()
            filename = f'image-{now}.png'
            logger.info(f'Payload received for topic "{msg.payload}", '
                        f'saving frame as: "{filename}"')
            # need this to access to the frame from within mqtt callback
            nonlocal frame
            cv2.imwrite(filename, frame)

        def start_record_cb(client, userdata, msg):
            session_state.record = True
            session_state.refresh = True

        def stop_record_cb(client, userdata, msg):
            session_state.record = False
            session_state.refresh = True

        conf = MQTTConfig()

        st.sidebar.markdown("___")
        st.sidebar.subheader("MQTT Options")
        selected_qos = st.sidebar.radio(
            'MQTT QoS', (0, 1, 2), deploy_conf.mqtt_qos, key='selected_qos')
        deploy_conf.mqtt_qos = int(selected_qos)
        st.sidebar.info(
            "#### Publishing Results to MQTT Topic:  \n"
            f"{conf.topics.publish_results}"
        )
        st.sidebar.info(
            "#### Subscribed MQTT Topics:  \n"
            f"**Start publishing results**: {conf.topics.start_publish}  \n"
            f"**Stop publishing results**: {conf.topics.stop_publish}  \n"
            f"**Save current frame**: {conf.topics.save_frame}  \n"
            f"**Start recording frames**: {conf.topics.start_record}  \n"
            f"**Stop recording frames**: {conf.topics.stop_record}")
        with st.sidebar.expander("Notes"):
            st.markdown("NOTE: Just publish an arbitrary message to any of the "
                        "subscribed MQTT topics to trigger the functionality.")

        if not session_state.client_connected:
            with st.spinner("Connecting to MQTT broker ..."):
                try:
                    session_state.client.connect(conf.broker, port=conf.port)
                except Exception as e:
                    st.error("Error connecting to MQTT broker")
                    logger.error(
                        f"Error connecting to MQTT broker {conf.broker}: {e}")
                    # return
                    st.stop()
                sleep(2)  # Wait for connection setup to complete
                logger.info("MQTT client connected successfully to "
                            f"{conf.broker} on port {conf.port}")
                session_state.client_connected = True

                # on_message will serve as fallback when none matched
                # or use this to be more precise on the subscription topic filter
                session_state.client.message_callback_add(
                    conf.topics.start_publish, start_publish_cb)
                session_state.client.message_callback_add(
                    conf.topics.stop_publish, stop_publish_cb)
                session_state.client.message_callback_add(
                    conf.topics.save_frame, save_frame_cb)
                session_state.client.message_callback_add(
                    conf.topics.start_record, start_record_cb)
                session_state.client.message_callback_add(
                    conf.topics.stop_record, stop_record_cb)
                for topic in conf.topics:
                    session_state.client.subscribe(topic, qos=selected_qos)
                session_state.client.loop_start()
                # need to add this to avoid Missing ReportContext error
                # https://github.com/streamlit/streamlit/issues/1326
                add_report_ctx(session_state.client._thread)

        # ************************ OUTPUT VIDEO STUFF ************************
        def stop_deployment():
            Deployment.reset_deployment_page()
            session_state.deployment_pagination = DeploymentPagination.Models
            st.experimental_rerun()

        st.subheader("Output Video")
        show_video_col = st.container()
        video_savepath_place = st.empty()
        record_text_place = st.empty()
        output_video_place = st.empty()
        publish_place = st.sidebar.empty()
        stop_deploy_col = st.sidebar.container()
        fps_col, width_col, height_col = st.columns(3)

        with show_video_col:
            show_video = st.checkbox(
                'Show video', value=True, key='show_video')
            draw_result = st.checkbox(
                "Draw labels", value=True, key='draw_result')
            if not session_state.record:
                video_savepath_place.empty()
                if st.button('Start recording', key='btn_start_record',
                             help="The video will be saved in your user 'Downloads' folder."):
                    session_state.record = True
                    st.experimental_rerun()
            else:
                stop_and_save_vid = st.button("Stop recording and save the video",
                                              key='btn_stop_and_save_vid')
                if stop_and_save_vid:
                    session_state.record = False
                    st.experimental_rerun()

        if session_state.publishing:
            if publish_place.button("Stop publishing results", key='btn_stop_pub'):
                session_state.publishing = False
                deploy_conf.publishing = False
                st.experimental_rerun()
        else:
            if publish_place.button("Start publishing results", key='btn_start_pub'):
                session_state.publishing = True
                deploy_conf.publishing = True
                st.experimental_rerun()

        with stop_deploy_col:
            st.button(
                "Stop deployment and reset", key='btn_stop_deploy',
                on_click=stop_deployment,
                help="""Please make sure to use this button to stop deployment before 
                proceeding to any other page!! Except when switching to another user,
                which is needed to prevent any other users from changing deployment settings.""")

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
            st.markdown("**Camera Frame Width**")
            st.markdown(kpi_format(width), unsafe_allow_html=True)
        with height_col:
            st.markdown("**Camera Frame Height**")
            st.markdown(kpi_format(height), unsafe_allow_html=True)
        st.markdown("___")

        inference_pipeline = session_state.deployment.get_inference_pipeline(
            draw_result=draw_result, **pipeline_kwargs)
        if DEPLOYMENT_TYPE == 'Semantic Segmentation with Polygons':
            get_result_fn = session_state.deployment.get_segmentation_results
        else:
            get_result_fn = partial(session_state.deployment.get_detection_results,
                                    conf_threshold=conf_threshold)
        publish_func = partial(session_state.client.publish,
                               conf.topics.publish_results, qos=selected_qos)

        if 'today' not in session_state:
            # using this to keep track of the current day for updating CSV file,
            # store in session_state to take into account the case when user
            # decided to move to another page during deployment
            session_state.today = datetime.now().date()

        # prepare CSV directory and path
        starting_time = datetime.now()
        csv_path = session_state.deployment.get_csv_path(starting_time)
        csv_dir = csv_path.parent
        os.makedirs(csv_dir, exist_ok=True)
        logger.info(f'Operation begins at: {starting_time.isoformat()}')
        logger.info(f'Inference results will be saved in {csv_dir}')
        with stop_deploy_col:
            with st.expander("CSV save file location"):
                st.markdown(f"**Inference results will be saved continuously in**: *{csv_dir}*  \n"
                            "A new file will be created daily. Be sure to click the `Stop deployment` "
                            "button to ensure the latest CSV file is saved properly if you need it.")

        fps = 0
        prev_time = 0
        first_csv_save = True
        while True:
            if session_state.refresh:
                # refresh page once to refresh the widgets
                logger.debug("REFRESHING PAGEE")
                session_state.refresh = False
                st.experimental_rerun()

            if use_cam:
                frame = session_state.camera.read()
            else:
                ret, frame = session_state.camera.read()
                if not ret:
                    break

            # frame.flags.writeable = True  # might need this?
            # run inference on the frame
            result = inference_pipeline(frame)
            if draw_result:
                output_img, pred = result
            else:
                output_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pred = result
            results = get_result_fn(pred)

            if show_video:
                output_video_place.image(output_img, width=selected_width)

            if session_state.record:
                # need to be within the video loop to ensure we also get the latest
                #  session_state updates from MQTT callback
                record_text_place.info("Recording ...")
                with show_video_col:
                    create_video_writer_if_not_exists()
                session_state.vid_writer.write(
                    cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
            else:
                record_text_place.empty()
                if session_state.vid_writer:
                    # remove text of the save location
                    video_savepath_place.empty()
                    logger.info("Saving recorded file")
                    # must release to close the video file
                    session_state.vid_writer.release()
                    session_state.vid_writer = None
                    st.experimental_rerun()

            # count FPS
            curr_time = perf_counter()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            fps_place.markdown(kpi_format(int(fps)), unsafe_allow_html=True)

            if show_labels:
                result_place.table(results)

            if not results:
                continue

            if session_state.publishing:
                payload = json.dumps(results)
                info = publish_func(payload=payload)

            # save results to CSV file
            if first_csv_save:
                first_csv_save = False
                if not csv_path.exists():
                    new_file = True
                else:
                    new_file = False
                create_csv_file_and_writer(
                    csv_path, results, new_file=new_file)
            now = datetime.now()
            today = now.date()
            if today > session_state.today:
                session_state.csv_file.close()
                csv_path = session_state.deployment.get_csv_path(now)
                session_state.today = today
                create_csv_file_and_writer(csv_path, results)
                st.experimental_rerun()
            else:
                for row in results:
                    session_state.csv_writer.writerow(row)

            # This below does not seem to work properly
            # if fps > max_allowed_fps:
            #     sleep(1 / (fps - max_allowed_fps))

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
