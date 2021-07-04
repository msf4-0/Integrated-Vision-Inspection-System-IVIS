"""
Title: Object Detection TF2
Date: 1/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
Description: Object detection
"""
import numpy as np
import cv2
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.utils import label_map_util
import tensorflow as tf
# import urllib.request
# import tarfile
import os
from pathlib import Path
import sys
import logging
from time import perf_counter
# -------------

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
sys.path.insert(0, str(Path(SRC, 'lib')))  # ./lib
# print(sys.path[0])
sys.path.insert(0, str(Path(Path(__file__).parent, 'module')))

#--------------------Logger-------------------------#
FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
DATEFMT = '%d-%b-%y %H:%M:%S'

# logging.basicConfig(filename='test.log',filemode='w',format=FORMAT, level=logging.DEBUG)
logging.basicConfig(format=FORMAT, level=logging.INFO,
                    stream=sys.stdout, datefmt=DATEFMT)

log = logging.getLogger()

#----------------------------------------------------#

'''
DATA_DIR = os.path.join(os.getcwd(), 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
for dir in [DATA_DIR, MODELS_DIR]:
    if not os.path.exists(dir):
        os.mkdir(dir)
'''
# ADD YOUR DIRECTORY HERE!!!!! 
DATA_DIR = '/home/chuzh/Documents/TensorFlow/workspace/training_demo'
MODELS_DIR = '/home/rchuzh/Documents/aruco/exported-models/model_2'

PATH_TO_CKPT = os.path.join(MODELS_DIR, 'checkpoint/')
PATH_TO_CFG = os.path.join(MODELS_DIR, 'pipeline.config')

LABEL_FILENAME = 'labelmap.pbtxt'
PATH_TO_LABELS = os.path.join(MODELS_DIR, LABEL_FILENAME)

#---------------------- Load the model -------------------------#

load_start_time = perf_counter()  # program loading start timestamp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(
    model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()


@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])


#--------------- Load label map data (for plotting)---------------------#
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

load_end_time = perf_counter()  # program loading end timestamp
load_time = load_start_time - load_end_time
# computes loading time of the program
log.info(f"Program Loading time = {load_time}")
# -----------------------Start Video Capture--------------------------#
link = "http://192.168.1.105:4747/video"  # IP webcam
local_link = "http://127.0.0.1:4747"    # not use -> /dev/video2
cap = cv2.VideoCapture(1)

while True:
    # Read frame from camera
    ret, image_np = cap.read()

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes']
         [0].numpy() + label_id_offset).astype(int),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)

    # Display output
    # cv2.imshow('object detection', cv2.resize(
    #     image_np_with_detections, (1280, 960)))
    cv2.imshow('object detection', image_np_with_detections)

    # QUIT PROGRAM
    # *********************************************
    # need to add a flag from Streamlit
    key = cv2.waitKey(1)

    ESC_KEY = 27
    # Quit.
    if key in {ord('q'), ord('Q'), ESC_KEY}:
        break
    # *********************************************

cap.release()
cv2.destroyAllWindows()
