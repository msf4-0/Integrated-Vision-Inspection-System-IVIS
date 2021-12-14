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

from collections import Counter
from datetime import datetime
from functools import partial
import json
import os
import shutil
import sys
from time import perf_counter, sleep
from typing import Callable

import cv2
from imutils.video.webcamvideostream import WebcamVideoStream
from matplotlib import pyplot as plt
import numpy as np
from paho.mqtt.client import Client
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state
from streamlit.report_thread import add_report_ctx
from project.project_management import Project

from user.user_management import User, UserRole

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>
# DEFINE Web APP page configuration
# layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
# LIB_PATH = SRC / "lib"
# if str(LIB_PATH) not in sys.path:
#     sys.path.insert(0, str(LIB_PATH))  # ./lib

from path_desc import TEMP_DIR, chdir_root
from core.utils.log import logger
from data_manager.database_manager import init_connection
from machine_learning.visuals import create_class_colors, create_color_legend
from data_manager.dataset_management import Dataset
from deployment.deployment_management import DeploymentConfig, DeploymentPagination, DeploymentType, Deployment
from deployment.utils import MQTTConfig, MQTTTopics, create_csv_file_and_writer, encode_frame, get_mqtt_client, reset_camera, reset_camera_ports, reset_csv_file_and_writer, reset_record_and_vid_writer
from core.utils.helper import Timer, get_all_timezones, get_now_string, list_available_cameras, save_captured_frame
from dobot_arm_demo import main as dobot_demo

# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


def kpi_format(text: str):
    return f"<h1 style='text-align: center; color:red;'>{text}</h1>"


def index(RELEASE=True):
    if 'deployment' not in session_state:
        st.warning("You have not deployed any model yet.")
        st.warning(
            "Please go to the Model Selection page and deploy a model first.")
        st.stop()

    if 'camera' not in session_state:
        session_state.camera = None
    if 'working_ports' not in session_state:
        session_state.working_ports = None
    if 'deployment_conf' not in session_state:
        # to store the config in case the user needs to go to another page during deployment
        session_state.deployment_conf = DeploymentConfig()

    deploy_conf: DeploymentConfig = session_state.deployment_conf

    project: Project = session_state.project
    deployment: Deployment = session_state.deployment
    user: User = session_state.user
    DEPLOYMENT_TYPE = deployment.deployment_type

    def stop_deployment():
        Deployment.reset_deployment_page()
        session_state.deployment_pagination = DeploymentPagination.Models

    def update_deploy_conf(conf_attr: str):
        """Update deployment config on any change of the widgets.

        NOTE: `conf_attr` must exist in the `session_state` (usually is the widget's state)
        and must be the same with the `DeploymentConfig` attribute's name."""
        val = session_state[conf_attr]
        logger.debug(f"Updated deploy_conf: {conf_attr} = {val}")
        setattr(deploy_conf, conf_attr, val)

    deploy_status_col = st.container()

    with deploy_status_col:
        st.subheader("Deployment Status")

        deploy_status_place = st.empty()
        deploy_status_place.info("**Status**: Not deployed. Please upload an image/video "
                                 "or use video camera.")

        st.warning("**NOTE**: Please do not simply refresh the page without ending "
                   "deployment or errors could occur.")

        deploy_btn_place = st.empty()
        pause_deploy_place = st.empty()

        st.button(
            "End Deployment", key='btn_stop_image_deploy',
            on_click=stop_deployment,
            help="This will stop the deployment and reset the entire  \n"
            "deployment configuration. Please **make sure** to use this  \n"
            "button to stop deployment before proceeding to any other  \n"
            "page! Or use the **pause button** (only available for camera  \n"
            "input) if you want to pause the deployment to do any other  \n"
            "things without resetting, such as switching user, or viewing  \n"
            "the latest saved CSV file (only for video camera deployment).")
        st.markdown("___")

    if user.role <= UserRole.Developer1:
        # use this variable to know whether the user has access to edit deployment config
        has_access = True
    else:
        has_access = False
        st.info(f"NOTE: Your user role **{user.role.fullname}** "
                "does not have access to editing deployment configuration.")
        st.markdown("___")

    st.sidebar.subheader("Configuration")

    if has_access:
        all_timezones = get_all_timezones()
        tz_idx = all_timezones.index(deploy_conf.timezone)
        st.sidebar.selectbox(
            "Local Timezone", all_timezones, index=tz_idx, key='timezone',
            help="Select your local timezone to have the correct time output in results.",
            on_change=update_deploy_conf, args=('timezone',))

        options = ('Image', 'Video')
        idx = options.index(deploy_conf.input_type)
        st.sidebar.radio(
            'Choose the Type of Input', options,
            index=idx, key='input_type',
            on_change=update_deploy_conf, args=('input_type',))
    else:
        st.info(f"Selected timezone: **{deploy_conf.timezone}**")
        st.markdown(f"**Input type**: {deploy_conf.input_type}")

    options_col, _ = st.columns(2)

    if DEPLOYMENT_TYPE == 'Image Classification':
        pipeline_kwargs = {}
    elif DEPLOYMENT_TYPE == 'Semantic Segmentation with Polygons':
        class_colors = create_class_colors(deployment.class_names)
        ignore_background = st.sidebar.checkbox(
            "Ignore background", value=deploy_conf.ignore_background,
            key='ignore_background',
            help="Ignore background class for visualization purposes.  \n"
            "Note that turning this on could significantly reduce the FPS.",
            on_change=update_deploy_conf, args=('ignore_background',))
        legend = create_color_legend(
            class_colors, bgr2rgb=False, ignore_background=ignore_background)
        st.sidebar.markdown("**Legend**")
        st.sidebar.image(legend)
        # convert to array
        class_colors = np.array(list(class_colors.values()),
                                dtype=np.uint8)
        pipeline_kwargs = {'class_colors': class_colors,
                           'ignore_background': ignore_background}
    else:
        if has_access:
            options_col.slider(
                "Confidence threshold:",
                min_value=0.1,
                max_value=0.99,
                value=deploy_conf.confidence_threshold,
                step=0.01,
                format='%.2f',
                key='confidence_threshold',
                help=("If a prediction's confidence score exceeds this threshold, "
                      "then it will be displayed, otherwise discarded."),
                on_change=update_deploy_conf, args=('confidence_threshold',)
            )
        else:
            options_col.markdown(
                f"**Confidence threshold**: {deploy_conf.confidence_threshold}")
        pipeline_kwargs = {'conf_threshold': deploy_conf.confidence_threshold}

    if deploy_conf.input_type == 'Image':
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
            display_width = st.sidebar.slider(
                "Select width of image to resize for display",
                35, 1000, 500, 5, key='display_width')
            # help=f'Original image width is **{ori_image_width}**.')
            st.sidebar.markdown(f"Original image width: **{ori_image_width}**")

        inference_pipeline = deployment.get_inference_pipeline(
            draw_result=True, **pipeline_kwargs)

        deploy_status_place.info("**Status**: Deployed for input images")

        if DEPLOYMENT_TYPE == 'Semantic Segmentation with Polygons':
            with st.expander("Notes about deployment for semantic segmentation"):
                st.markdown(
                    """If this is an externally trained segmentation model, it might 
                    not work properly for making predictions for our app's implementation.
                    Please try with sample images or uploaded images first to see the results. 
                    If there is anything wrong, please do not proceed to video deployment.""")

        with st.spinner("Running inference ..."):
            try:
                with Timer("Inference on image"):
                    result = inference_pipeline(img)
            except Exception as e:
                # uncomment the following line to see the traceback
                # st.exception(e)
                logger.error(
                    f"Error running inference with the model: {e}")
                st.error("""Error when trying to run inference with the model,
                    please check with Admin/Developer for debugging.""")
                st.stop()
        if DEPLOYMENT_TYPE == 'Image Classification':
            pred_classname, y_proba = result
            caption = (f"{filename}; "
                       f"Predicted: {pred_classname}; "
                       f"Score: {y_proba * 100:.1f}")
            st.image(img, channels='BGR', width=display_width, caption=caption)
        elif DEPLOYMENT_TYPE == 'Semantic Segmentation with Polygons':
            drawn_mask_output, _ = result
            # convert to RGB for visualizing with Matplotlib
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            st.subheader(filename)
            fig = plt.figure()
            plt.subplot(121)
            plt.title("Original Image")
            plt.imshow(rgb_img)
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
            st.image(img_with_detections, width=display_width,
                     caption=f'Detection result for: {filename}')

        st.stop()

    elif deploy_conf.input_type == 'Video':
        def update_conf_and_reset_camera(conf_attr: str):
            update_deploy_conf(conf_attr)
            reset_camera()

        # Does not seem to work properly
        # max_allowed_fps = st.sidebar.slider(
        #     'Maximum frame rate', 1, 60, 24, 1,
        #     key='selected_max_allowed_fps', on_change=reset_camera,
        #     help="""This is the maximum allowed frame rate that the
        #         videostream will run at.""")
        if has_access:
            st.sidebar.slider(
                'Width of video', 320, 1920, deploy_conf.video_width, 10,
                key='video_width',
                help="This is the width of video for visualization purpose.",
                on_change=update_deploy_conf, args=('video_width',)
            )

            st.sidebar.checkbox(
                'Use video camera', value=deploy_conf.use_camera,
                key='use_camera',
                on_change=update_conf_and_reset_camera, args=('use_camera',))
        else:
            st.sidebar.markdown(
                f"**Width of video**: {deploy_conf.video_width}")
            if deploy_conf.use_camera:
                st.sidebar.markdown("Using **video camera** for deployment.")
            else:
                st.sidebar.markdown("Using **uploaded video**.")

        if deploy_conf.use_camera:
            # TODO: test using streamlit-webrtc

            with st.sidebar.container():
                if has_access:
                    options = ('USB Camera', 'IP Camera')
                    idx = options.index(deploy_conf.camera_type)
                    st.radio(
                        "Select type of camera", options, index=idx,
                        key='camera_type', on_change=update_conf_and_reset_camera,
                        args=('camera_type',))
                if deploy_conf.camera_type == 'USB Camera':
                    if has_access:
                        st.button("Refresh camera ports", key='btn_refresh',
                                  on_click=reset_camera_ports)

                        if not session_state.working_ports:
                            with st.spinner("Checking available camera ports ..."):
                                _, working_ports = list_available_cameras()
                                session_state.working_ports = working_ports.copy()
                            # stop if no camera port found
                            if not working_ports:
                                st.error(
                                    "No working camera source/port found.")
                                logger.error("No working camera port found")
                                st.stop()

                        st.radio(
                            "Select a camera port",
                            options=session_state.working_ports,
                            index=deploy_conf.camera_port,
                            key='camera_port',
                            on_change=update_conf_and_reset_camera,
                            args=('camera_port',))
                    else:
                        st.markdown("USB Camera from camera port: "
                                    f"**{deploy_conf.camera_port}**")
                    video_source = deploy_conf.camera_port
                else:
                    if has_access:
                        st.text_input(
                            "Enter the IP address", value=deploy_conf.ip_cam_address,
                            key='ip_cam_address',
                            on_change=update_deploy_conf, args=(
                                'ip_cam_address',),
                            help="""This address could start with *http* or *rtsp*.
                            Most of the IP cameras  \nhave a username and password to access
                            the video. In such case,  \nthe credentials have to be provided
                            in the streaming URL as follow:  \n
                            **rtsp://username:password@192.168.1.64/1**""")
                    else:
                        st.markdown("IP Camera with address: "
                                    f"**{deploy_conf.ip_cam_address}** ")
                    video_source = deploy_conf.ip_cam_address

            # **************************** CSV FILE STUFF ****************************
            if 'today' not in session_state:
                # using this to keep track of the current day for updating CSV file,
                # store in session_state to take into account the case when user
                # decided to move to another page during deployment
                session_state.today = datetime.now().date()

            def update_retention_period():
                retention_period = session_state.day_input \
                    + (7 * session_state.week_input) \
                    + (30 * session_state.month_input)
                if retention_period == 0:
                    warning_place.warning(
                        "Retention period must be larger than 1 day!")
                    return
                # in 'days' unit
                deploy_conf.retention_period = retention_period

            # show CSV directory
            csv_path = deployment.get_csv_path(datetime.now())
            csv_dir = csv_path.parents[1]
            logger.info(f'Inference results will be saved in {csv_dir}')
            with st.sidebar.container():
                st.markdown("___")
                st.subheader("Info about saving results")
                st.markdown("#### Data retention period")
                warning_place = st.empty()
                if has_access:
                    with st.form('retention_period_form', clear_on_submit=True):
                        st.number_input("Day", 0, 1000, deploy_conf.retention_period, 1,
                                        key='day_input')
                        st.number_input("Week", 0, 10, 0, 1, key='week_input',
                                        help='7 days per week')
                        st.number_input("Month", 0, 12, 0, 1, key='month_input',
                                        help='30 days per month')
                        st.form_submit_button(
                            'Change retention period', on_click=update_retention_period)
                retention_period = deploy_conf.retention_period

                st.markdown(f"Retention period = **{retention_period} days**")
                with st.expander("CSV save file info"):
                    st.markdown(
                        f"**Inference results will be saved continuously in**: *{csv_dir}*  \n"
                        f"A new file will be created daily and files older than the "
                        f"retention period (`{retention_period} days`) will be deleted. "
                        "Be sure to click the `Pause deployment` button "
                        "to ensure the latest CSV file is saved properly if you have any "
                        "problem with opening the file.")
        else:
            video_file = st.sidebar.file_uploader(
                "Or upload a video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'],
                key='video_file_uploader', on_change=reset_camera)

        # **************************** MQTT STUFF ****************************
        saved_frame_dir = deployment.get_frame_save_dir('image')
        recording_dir = deployment.get_frame_save_dir('video')
        for save_dir in (saved_frame_dir, recording_dir):
            if not save_dir.exists():
                os.makedirs(save_dir)

        if 'client' not in session_state:
            # NOTE: using project ID as the client ID for now
            session_state.client = get_mqtt_client(str(project.id))
            session_state.client_connected = False
            session_state.publishing = True

            session_state.mqtt_conf = MQTTConfig()

        if 'csv_writer' not in session_state:
            session_state.csv_writer = None
            session_state.csv_file = None

            # use this to refresh the page once to show widget changes
            session_state.refresh = False

        if 'record' not in session_state:
            session_state.record = False
            session_state.vid_writer = None

        def start_publish_cb(client, userdata, msg):
            logger.info("Start publishing")
            session_state.publishing = True
            deploy_conf.publishing = True
            session_state.refresh = True

        def stop_publish_cb(client, userdata, msg):
            logger.info("Stopping publishing ...")
            # session_state.client_connected = False
            session_state.publishing = False
            deploy_conf.publishing = False
            deploy_conf.publish_frame = False
            logger.info("Stopped")
            session_state.refresh = True
            # st.success("Stopped publishing")
            # cannot use this within mqtt callback
            # st.experimental_rerun()

        def start_publish_frame_cb(client, userdata, msg):
            logger.info("Start publishing frames")
            deploy_conf.publish_frame = True
            session_state.refresh = True

        def stop_publish_frame_cb(client, userdata, msg):
            logger.info("Stopping publishing frames...")
            deploy_conf.publish_frame = False
            logger.info("Stopped")
            session_state.refresh = True

        def save_frame_cb(client, userdata, msg):
            logger.debug(f'Payload received for topic "{msg.topic}"')
            # need this to access to the frame from within mqtt callback
            nonlocal output_img
            save_captured_frame(output_img, saved_frame_dir)

        def start_record_cb(client, userdata, msg):
            session_state.record = True
            session_state.refresh = True

        def stop_record_cb(client, userdata, msg):
            session_state.record = False
            session_state.refresh = True

        client: Client = session_state.client
        conf: MQTTConfig = session_state.mqtt_conf

        def add_client_callbacks():
            topics: MQTTTopics = session_state.mqtt_conf.topics

            # on_message() will serve as fallback when none matched
            # or use this to be more precise on the subscription topic filter
            client.message_callback_add(
                topics.start_publish, start_publish_cb)
            client.message_callback_add(
                topics.stop_publish, stop_publish_cb)
            client.message_callback_add(
                topics.start_publish_frame, start_publish_frame_cb)
            client.message_callback_add(
                topics.stop_publish_frame, stop_publish_frame_cb)
            client.message_callback_add(
                topics.save_frame, save_frame_cb)
            client.message_callback_add(
                topics.start_record, start_record_cb)
            client.message_callback_add(
                topics.stop_record, stop_record_cb)

        st.sidebar.markdown("___")
        st.sidebar.subheader("MQTT Options")

        # connect MQTT broker and set up callbacks
        if not session_state.client_connected:
            logger.debug(f"{conf = }")
            logger.debug(f"{conf.topics = }")
            with st.spinner("Connecting to MQTT broker ..."):
                try:
                    client.connect(conf.broker, port=conf.port)
                except Exception as e:
                    st.error("Error connecting to MQTT broker")
                    # st.exception(e)
                    logger.error(
                        f"Error connecting to MQTT broker {conf.broker}: {e}")
                    # return
                    st.stop()

                sleep(2)  # Wait for connection setup to complete
                logger.info("MQTT client connected successfully to "
                            f"{conf.broker} on port {conf.port}")
                session_state.client_connected = True

                add_client_callbacks()

                for topic in conf.topics.__dict__.values():
                    client.subscribe(topic, qos=conf.qos)

                client.loop_start()

                # need to add this to avoid Missing ReportContext error
                # https://github.com/streamlit/streamlit/issues/1326
                add_report_ctx(client._thread)

        # NOTE: Docker needs to use service name instead to connect to broker,
        # but user should always connect to 'localhost'
        st.sidebar.info(f"**MQTT broker**: localhost  \n**Port**: {conf.port}")

        if has_access:
            topic_error_place = st.sidebar.empty()

            def update_mqtt_qos():
                # take from the widget's state and save to our mqtt_conf
                logger.info(f"Updated QoS level from {conf.qos} to "
                            f"{session_state.mqtt_qos}")
                conf.qos = session_state.mqtt_qos

                for topic in conf.topics.__dict__.values():
                    client.unsubscribe(topic)
                    client.subscribe(topic, qos=conf.qos)

            # def update_conf_topic(topic_attr: str):
            def update_conf_topic():
                for topic_attr in conf.topics.__dict__.keys():
                    new_topic = session_state[topic_attr]
                    if new_topic == '':
                        logger.error('Topic cannot be empty string')
                        topic_error_place.error('Topic cannot be empty string')
                        sleep(1)
                        st.experimental_rerun()

                    previous_topic = getattr(conf.topics, topic_attr)

                    if new_topic == previous_topic:
                        # no need to change anything if user didn't change the topic
                        continue

                    # unsubscribe the old topic and remove old topic callback
                    client.unsubscribe(previous_topic)
                    client.message_callback_remove(previous_topic)

                    # update MQTTConfig with new topic, add callback, and subscribe
                    setattr(conf.topics, topic_attr, new_topic)

                    if topic_attr not in ('publish_results', 'publish_frame'):
                        # only these two topics don't have callbacks to add
                        callback_func = eval(f'{topic_attr}_cb')
                        client.message_callback_add(new_topic, callback_func)
                    client.subscribe(new_topic, qos=conf.qos)

                    logger.info(f"Updated MQTTConfig.{topic_attr} from {previous_topic} "
                                f"to {new_topic}")

            st.sidebar.radio(
                'MQTT QoS', (0, 1, 2), conf.qos, key='mqtt_qos',
                on_change=update_mqtt_qos)

            st.sidebar.markdown("**MQTT Topics**")
            st.sidebar.markdown(
                "If you change any MQTT topic name(s), please click the **Update Config** "
                "button to allow the changes to be made.")
            # must clear on submit to show the correct values on form
            with st.sidebar.form('form_mqtt_topics', clear_on_submit=True):
                st.text_input(
                    'Publishing Results to MQTT Topic', conf.topics.publish_results,
                    key='publish_results')
                st.text_input(
                    'Publishing output frames to', conf.topics.publish_frame,
                    key='publish_frame')
                st.text_input(
                    'Start publishing results', conf.topics.start_publish,
                    key='start_publish')
                st.text_input(
                    'Stop publishing results', conf.topics.stop_publish,
                    key='stop_publish')
                st.text_input(
                    'Start publishing frames', conf.topics.start_publish_frame,
                    key='start_publish_frame')
                st.text_input(
                    'Stop publishing frames', conf.topics.stop_publish_frame,
                    key='stop_publish_frame')
                st.text_input(
                    'Save current frame', conf.topics.save_frame,
                    key='save_frame')
                st.text_input(
                    'Start recording frames', conf.topics.start_record,
                    key='start_record')
                st.text_input(
                    'Stop recording frames', conf.topics.stop_record,
                    key='stop_record')
                st.form_submit_button(
                    "Update Config", on_click=update_conf_topic,
                    help="Please press this button to update if you change any MQTT "
                    "topic name(s).")
        else:
            st.sidebar.markdown(
                f"MQTT QoS is set to level **{conf.qos}**")
            st.sidebar.info(
                "#### Publishing Results to MQTT Topic:  \n"
                f"{conf.topics.publish_results}  \n"
                "#### Publishing output frames to:  \n"
                f"{conf.topics.publish_frame}"
            )
            st.sidebar.info(
                "#### Subscribed MQTT Topics:  \n"
                f"**Start publishing results**: {conf.topics.start_publish}  \n"
                f"**Stop publishing results**: {conf.topics.stop_publish}  \n"
                f"**Start publishing frames**: {conf.topics.start_publish_frame}  \n"
                f"**Stop publishing frames**: {conf.topics.stop_publish_frame}  \n"
                f"**Save current frame**: {conf.topics.save_frame}  \n"
                f"**Start recording frames**: {conf.topics.start_record}  \n"
                f"**Stop recording frames**: {conf.topics.stop_record}")
        with st.sidebar.expander("Notes"):
            st.markdown(
                f"Make sure to connect to **'localhost'** broker (or IP Address of this PC) "
                f"with the correct port **{conf.port}**. "
                "Then just publish an arbitrary message to any of "
                "the subscribed MQTT topics to trigger the functionality.  \nFor the **saved "
                f"frames**, they will be saved in your project's folder at *{saved_frame_dir}*, "
                f"while recorded video will be saved at *{recording_dir}*. Please "
                "**do not simply delete these folders during deployment**, "
                "otherwise error will occur. You can delete them after pausing/ending "
                "the deployment if you wish.  \n"
                "For the **publishing frames**, they are in **bytes** format.")

        # ************************ Video deployment button ************************

        # allow the user to click the "Deploy Model button" after done configuring everything
        if deploy_conf.input_type == 'Video':
            if deploy_conf.use_camera:
                # only show these if a camera is not selected and not deployed yet
                if not session_state.camera:
                    if deploy_btn_place.button(
                        "🛠️ Deploy Model", key='btn_deploy_cam',
                            help='Deploy your model with the selected camera source'):
                        with st.spinner("Loading up camera ..."):
                            # NOTE: VideoStream does not work with filepath
                            try:
                                session_state.camera = WebcamVideoStream(
                                    src=video_source).start()
                                if session_state.camera.read() is None:
                                    raise Exception(
                                        "Video source is not valid")
                            except Exception as e:
                                st.error(
                                    f"Unable to read from video source {video_source}")
                                logger.error(
                                    f"Unable to read from video source {video_source}: {e}")
                                st.stop()

                            sleep(2)  # give the camera some time to sink in
                            # rerun just to avoid displaying unnecessary buttons
                            st.experimental_rerun()
            else:
                if not video_file:
                    st.stop()

                video_path = str(TEMP_DIR / video_file.name)

                if deploy_btn_place.button(
                    "🛠️ Deploy Model", key='btn_deploy_cam',
                        help='Deploy your model with the selected camera source'):

                    if TEMP_DIR.exists():
                        shutil.rmtree(TEMP_DIR)
                    os.makedirs(TEMP_DIR)
                    logger.debug(f"{video_path = }")
                    with open(video_path, 'wb') as f:
                        f.write(video_file.getvalue())

                    try:
                        session_state.camera = cv2.VideoCapture(video_path)
                        assert session_state.camera.isOpened(), "Video is unreadable"
                    except Exception as e:
                        st.error(f"Unable to read from the video file: "
                                 f"'{video_file.name}'")
                        logger.error(f"Unable to read from the video file: "
                                     f"'{video_file.name}' with error: {e}")
                        st.stop()

                    sleep(2)  # give the camera some time to sink in
                    st.experimental_rerun()

        # after user has clicked the "Deploy Model button"
        if session_state.camera:
            if isinstance(session_state.camera, WebcamVideoStream):
                stream = session_state.camera.stream
                deploy_status_place.info(
                    "**Status**: Deployed for video camera input")
            else:
                stream = session_state.camera
                deploy_status_place.info(
                    "**Status**: Deployed for uploaded video")

                with deploy_status_col:
                    st.subheader('Input Video')
                    try:
                        st.video(video_path)
                    except Exception as e:
                        st.error(f"Unable to read from the video file: "
                                 f"'{video_file.name}'")
                        logger.error(f"Unable to read from the video file: "
                                     f"'{video_file.name}' with error: {e}")
                        st.stop()

            width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_input = int(stream.get(cv2.CAP_PROP_FPS))
            logger.info(
                f"Video properties: {width = }, {height = }, {fps_input = }")

            def pause_deployment():
                reset_camera()
                reset_record_and_vid_writer()
                reset_csv_file_and_writer()

            pause_deploy_place.button(
                "Pause Deployment", key='btn_pause_deploy',
                on_click=pause_deployment,
                help=("Pause deployment after you have deployed the model  \n"
                      "with a running video camera. Or use this to reset  \n"
                      "camera if there is any problem with loading up  \n"
                      "the camera. Note that this is extremely important  \n"
                      "to ensure your camera is properly stopped and the  \n"
                      "camera access is given back to your system. This will  \n"
                      "also save the latest CSV file in order to be opened."))
        else:
            st.stop()

        # *********************** DOBOT arm demo ***********************
        if 'check_labels' not in session_state:
            session_state.check_labels = None

            def get_current_box_view(client, userdata, msg):
                view: str = msg.payload.decode()
                logger.info(f"Received message from topic '{msg.topic}': "
                            f"'{view}'")
                session_state.check_labels = view

            client.subscribe(conf.topics.dobot_view)
            client.message_callback_add(
                conf.topics.dobot_view, get_current_box_view)

        # DOBOT_TASK = dobot_demo.DobotTask.Box  # for box shapes
        DOBOT_TASK = dobot_demo.DobotTask.P2_143  # for machine part

        if DOBOT_TASK == dobot_demo.DobotTask.Box:
            VIEW_LABELS = dobot_demo.BOX_VIEW_LABELS
        elif DOBOT_TASK == dobot_demo.DobotTask.P2_143:
            VIEW_LABELS = dobot_demo.P2_143_VIEW_LABELS

        st.button("Move DOBOT and detect",
                  key='btn_move_dobot', on_click=dobot_demo.run,
                  args=(conf, conf.topics.dobot_view, conf.qos,
                        DOBOT_TASK))

        # *********************** Deployment video loop ***********************
        def create_video_writer_if_not_exists():
            if not session_state.vid_writer:
                logger.info("Creating video file to record to")
                # NOTE: THIS VIDEO SAVE FORMAT IS VERY PLATFORM DEPENDENT
                # TODO: THE VIDEO FILE MIGHT NOT SAVE PROPERLY
                # usually either MJPG + .avi, or XVID + .mp4
                FOURCC = cv2.VideoWriter_fourcc(*"XVID")
                filename = f"video_{get_now_string(timezone=deploy_conf.timezone)}.mp4"
                video_save_path = str(recording_dir / filename)
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

        st.subheader("Output Video")
        show_video_col = st.container()
        msg_cont, _ = st.columns(2)
        with msg_cont:
            msg_place = st.empty()
        output_video_place = st.empty()
        publish_place = st.sidebar.empty()
        fps_col, width_col, height_col = st.columns(3)

        with show_video_col:
            show_video = st.checkbox(
                'Show video', value=True, key='show_video')
            draw_result = st.checkbox(
                "Draw labels", value=True, key='draw_result')

            def update_record(is_recording: bool):
                session_state.record = is_recording

            if not session_state.record:
                st.button('Start recording', key='btn_start_record',
                          help=f"The video will be saved in *{recording_dir}*",
                          on_click=update_record, args=(True,))
            else:
                st.button("Stop recording and save the video",
                          key='btn_stop_and_save_vid',
                          on_click=update_record, args=(False,))

        if has_access:
            def update_publishing_conf(is_publishing: bool):
                deploy_conf.publishing = is_publishing
                session_state.publishing = is_publishing
                # also change whether publishing frame or not
                deploy_conf.publish_frame = is_publishing

            def update_publish_frame_conf(publish_frame: bool):
                deploy_conf.publish_frame = publish_frame

            if session_state.publishing:
                # using buttons to allow the widget to change after rerun
                # whereas checkbox does not change after rerun
                publish_place.button("Stop publishing results", key='btn_stop_pub',
                                     on_click=update_publishing_conf, args=(False,))
            else:
                publish_place.button("Start publishing results", key='btn_start_pub',
                                     on_click=update_publishing_conf, args=(True,))

            if deploy_conf.publish_frame:
                st.sidebar.button(
                    "Stop publishing frames", key='btn_stop_pub_frame',
                    help="Stop publishing frames as bytes to the MQTT Topic: "
                    f"*{conf.topics.publish_frame}*.",
                    on_click=update_publish_frame_conf, args=(False,))
            else:
                st.sidebar.button(
                    "Start publishing frames", key='btn_start_pub_frame',
                    help="Publish frames as bytes to the MQTT Topic:  \n"
                    f"*{conf.topics.publish_frame}*.  \nNote that this could significantly "
                    "reduce FPS.", on_click=update_publish_frame_conf, args=(True,))

        else:
            session_state.publishing = deploy_conf.publishing
            if session_state.publishing:
                st.markdown("Currently is publishing results to the topic: "
                            f"*{conf.topics.publish_results}*")
            else:
                st.markdown(
                    "Currently is not publishing any results through MQTT.")

            if deploy_conf.publish_frame:
                st.markdown("Currently is publishing output frames to the topic: "
                            f"*{conf.topics.publish_frame}*")
            else:
                st.markdown(
                    "Currently is not publishing any output frames through MQTT.")

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

        # prepare variables for the video deployment loop
        inference_pipeline = deployment.get_inference_pipeline(
            draw_result=draw_result, **pipeline_kwargs)

        if DEPLOYMENT_TYPE == 'Image Classification':
            is_image_classif = True
            get_result_fn = partial(deployment.get_classification_results,
                                    timezone=deploy_conf.timezone)
        elif DEPLOYMENT_TYPE == 'Semantic Segmentation with Polygons':
            is_image_classif = False
            get_result_fn = partial(deployment.get_segmentation_results,
                                    timezone=deploy_conf.timezone)
        else:
            is_image_classif = False
            get_result_fn = partial(deployment.get_detection_results,
                                    conf_threshold=deploy_conf.confidence_threshold,
                                    timezone=deploy_conf.timezone)
        publish_func = partial(client.publish,
                               conf.topics.publish_results, qos=conf.qos)
        publish_frame_fn = partial(client.publish,
                                   conf.topics.publish_frame, qos=conf.qos)

        starting_time = datetime.now()
        csv_path = deployment.get_csv_path(starting_time)
        csv_dir = csv_path.parent
        if not csv_dir.exists():
            os.makedirs(csv_dir)
        logger.info(f'Operation begins at: {starting_time.isoformat()}')
        logger.info(f'Inference results will be saved in {csv_dir}')
        use_cam = deploy_conf.use_camera
        display_width = deploy_conf.video_width
        publish_frame = deploy_conf.publish_frame

        fps = 0
        prev_time = 0
        first_csv_save = True
        # for DOBOT DEMO
        prev_view = None

        # start the video deployment loop
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
            if is_image_classif:
                results = get_result_fn(*result)
                output_img = frame
                # the read frame is in BGR format
                channels = 'BGR'
            else:
                if draw_result:
                    output_img, pred = result
                    channels = 'RGB'
                else:
                    pred = result
                    output_img = frame
                    # the read frame is in BGR format
                    channels = 'BGR'
                results = get_result_fn(pred)

            if show_video:
                output_video_place.image(output_img, channels=channels,
                                         width=display_width)

            if session_state.record:
                # need to be within the video loop to ensure we also get the latest
                #  session_state updates from MQTT callback
                msg_place.info("Recording ...")
                with show_video_col:
                    create_video_writer_if_not_exists()
                if channels == 'RGB':
                    # cv2.VideoWriter needs BGR format
                    out = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
                else:
                    out = output_img
                session_state.vid_writer.write(out)
            else:
                if session_state.vid_writer:
                    msg_place.empty()
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

            if publish_frame:
                if channels == 'RGB':
                    # OpenCV needs BGR format
                    out = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
                else:
                    out = output_img
                frame_bytes = encode_frame(out)
                info = publish_frame_fn(frame_bytes)

            # NOTE: this session_state is currently ONLY used for DOBOT arm demo for
            # shape detection on different views
            if session_state.check_labels:
                view: str = session_state.check_labels
                if view == 'end':
                    # clear the message if the robot motion is ended
                    msg_place.empty()
                    # and reset back to None
                    session_state.check_labels = None
                    continue
                if view == prev_view:
                    # ONLY CHECK FOR ONCE for the same view
                    continue

                required_label_cnts = VIEW_LABELS[view]
                # sort by label name
                required_label_cnts = sorted(
                    required_label_cnts.items(), key=lambda x: x[0])
                detected_labels = [r['name'] for r in results]
                detected_label_cnts = Counter(detected_labels)
                # sort by label name
                detected_label_cnts = sorted(
                    detected_label_cnts.items(), key=lambda x: x[0])
                logger.info(f"Required labels: {required_label_cnts}")
                logger.info(f"Detected labels: {detected_label_cnts}")
                if detected_label_cnts == required_label_cnts:
                    logger.info(f"All labels present at '{view}' view")
                    msg_place.success(f"### {view.upper()} view: OK")
                else:
                    logger.warning("Required labels are not detected at "
                                   f"'{view}' view")
                    msg_place.error(f"### {view.upper()} view: NG")

                prev_view = view

            if not results:
                continue

            if session_state.publishing:
                payload = json.dumps(results)
                info = publish_func(payload=payload)

            # save results to CSV file only if using video camera
            if not use_cam:
                continue

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

                deployment.delete_old_csv_files(
                    retention_period)

                csv_path = deployment.get_csv_path(now)
                session_state.today = today
                create_csv_file_and_writer(csv_path, results)
                st.experimental_rerun()
            else:
                for row in results:
                    session_state.csv_writer.writerow(row)

            # This below does not seem to work properly
            # if fps > max_allowed_fps:
            #     sleep(1 / (fps - max_allowed_fps))

        # clean up everything if it's an uploaded video
        if session_state.vid_writer:
            logger.info("Saving recorded file")
            # must release to close the video file
            session_state.vid_writer.release()
            session_state.vid_writer = None

        reset_camera()

        if TEMP_DIR.exists():
            logger.debug("Removing temporary directory")
            shutil.rmtree(TEMP_DIR)

        st.info("Inference done for uploaded video.")


def main():
    # False for debugging
    index(RELEASE=False)


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
