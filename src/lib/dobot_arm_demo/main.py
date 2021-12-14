from collections import Counter
from typing import Any, Dict, List, Tuple
from pathlib import Path
import sys
from enum import IntEnum

from paho.mqtt.client import Client

if __name__ == '__main__':
    from dobot_api import dobot_api_dashboard, dobot_api_feedback, MyType
else:
    from .dobot_api import dobot_api_dashboard, dobot_api_feedback, MyType

from threading import Thread
import numpy as np
import time

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

from core.utils.log import logger
from deployment.utils import MQTTConfig, get_mqtt_client


class DobotTask(IntEnum):
    Box = 0
    P2_143 = 1


# view -> (label, required_number_of_the_label)
BOX_VIEW_LABELS: Dict[str, List[Tuple[str, int]]] = {
    'top': [('hexagon', 3)],
    'back': [('Rectangle', 1)],
    'right': [('circle', 2)],
    'front': [('omron-sticker', 1)],
    'left': [('Triangle', 2)],
}

# machine part P2/143
P2_143_VIEW_LABELS: Dict[str, List[Tuple[str, int]]] = {
    'top': [('date', 1), ('white dot', 1)],
    'top left': [('top weld', 1)],
    'top right': [('top weld', 1)],
    'left': [('left weld', 1)],
    'right': [('right weld', 1)],
}


def check_result_labels(results: List[Dict[str, Any]], required_label_cnts: List[Tuple[str, int]],
                        check_subset: bool = True) -> bool:
    """Check the object detection `results` to see whether all `required_label_cnts` 
    are found in the `results`.

    `required_label_cnts` should be obtained from the `VIEW_LABELS` dictionary.

    `results` should be obtained from the deployment_page `get_result_fn()`.

    If `check_subset` is True, only check that the required labels (with the correct count)
    are present in the results, and disregard all the other detected labels that are not
    in the required labels. Else, all detected label counts must be equal to the required 
    label counts.
    """
    # sort by label name
    required_label_cnts = sorted(
        required_label_cnts.items(), key=lambda x: x[0])

    detected_labels: List[str] = [r['name'] for r in results]
    detected_label_cnts = Counter(detected_labels)
    # sort by label name
    detected_label_cnts = sorted(
        detected_label_cnts.items(), key=lambda x: x[0])

    logger.info(f"Required labels: {required_label_cnts}")
    logger.info(f"Detected labels: {detected_label_cnts}")

    if check_subset:
        if set(required_label_cnts).issubset(detected_label_cnts):
            return True
    else:
        if detected_label_cnts == required_label_cnts:
            return True
    return False


def move_for_box(client_feedback: dobot_api_feedback, client: Client, topic: str, qos: int):
    # move to origin (top view)
    client_feedback.JointMovJ(
        (0.05), (-38.74), (-118.19), (157.46), (87.44), (0))
    time.sleep(3)
    # send the current view as the payload to our vision inspection app
    client.publish(topic, 'top', qos)
    time.sleep(1)

    # move to back side
    client_feedback.JointMovJ(
        (0.54), (-50.16), (-153.78), (114.97), (89.54), (-178))
    # allow time for the dobot arm to move to the position
    time.sleep(8)
    client.publish(topic, 'back', qos)
    time.sleep(2)

    # move to right side
    client_feedback.JointMovJ(
        (27.478), (-47.836), (-111.595), (70.479), (89.5876), (-63.62))
    time.sleep(5)
    client.publish(topic, 'right', qos)
    time.sleep(2)

    # move to front side
    client_feedback.JointMovJ(
        (0.56), (-76.07), (-37.9), (25.09), (90.11), (-2.21))
    time.sleep(5)
    client.publish(topic, 'front', qos)
    time.sleep(2)

    # move to left side
    client_feedback.JointMovJ(
        (-24.19), (-59.51), (-79), (49.49), (90.57), (62.96))
    time.sleep(4)
    client.publish(topic, 'left', qos)
    time.sleep(2)

    # move back to origin (top view)
    client_feedback.JointMovJ(
        (0.05), (-38.74), (-118.19), (157.46), (87.44), (0))
    time.sleep(5)
    client.publish(topic, 'end', qos)


def move_for_p2_143(client_feedback: dobot_api_feedback, client: Client, topic: str, qos: int):
    # origin top
    client_feedback.JointMovJ(
        (0.0523), (-37.6843), (-121.325), (158.5393), (87.4376), (0))
    time.sleep(3)

    # top checking
    client_feedback.JointMovJ(
        (0.0539), (-50.2704), (-121.7125), (172.513), (87.4359), (0))
    time.sleep(5)
    client.publish(topic, 'top', qos)
    time.sleep(2)

    # Intermediate 1
    client_feedback.JointMovJ(
        (0.0589), (-49.7576), (-127.02), (177.3076), (87.4310), (0))
    time.sleep(2)

    # Top Right 1
    client_feedback.JointMovJ(
        (21.68), (-54.5190), (-121.4119), (172.052), (65.8095), (-32.8328))
    time.sleep(5)
    client.publish(topic, 'top right', qos)
    time.sleep(2)

    # Right 2
    client_feedback.JointMovJ(
        (19.2584), (-51.7799), (-106.9316), (67.7758), (82.393), (-59.3024))
    time.sleep(5)
    client.publish(topic, 'right', qos)
    time.sleep(2)

    # origin top
    client_feedback.JointMovJ(
        (0.0523), (-37.6843), (-121.325), (158.5393), (87.4376), (0))
    time.sleep(2)

    # Top Left 1
    client_feedback.JointMovJ(
        (-13.2228), (-40.179), (-95.0444), (50.8133), (58.3560), (70.5871))
    time.sleep(8)
    client.publish(topic, 'top left', qos)
    time.sleep(2)

    # Left 2
    client_feedback.JointMovJ(
        (-16.1938), (-62.0114), (-67.4206), (41.9222), (87.7707), (31.6594))
    time.sleep(5)
    client.publish(topic, 'left', qos)
    time.sleep(2)

    # Intermediate 2
    client_feedback.JointMovJ(
        (-16.1938), (-50.0761), (-64.2679), (26.8342), (87.7707), (31.6594))
    time.sleep(2)

    # origin top
    client_feedback.JointMovJ(
        (0.0523), (-37.6843), (-121.325), (158.5393), (87.4376), (0))
    time.sleep(5)
    client.publish(topic, 'end', qos)


def move_and_publish_view(client_dashboard: dobot_api_dashboard, client_feedback: dobot_api_feedback,
                          conf: MQTTConfig, task: DobotTask):
    topic = conf.topics.dobot_view
    qos = conf.qos

    client = get_mqtt_client()
    client.connect(conf.broker, port=conf.port)
    client.subscribe(topic, qos)
    client.loop_start()

    # Remove alarm
    client_dashboard.ClearError()
    time.sleep(0.5)
    # Description The upper function was enabled successfully
    client_dashboard.EnableRobot()
    time.sleep(0.5)

    if task == DobotTask.Box:
        move_for_box(client_feedback, client, topic, qos)
    elif task == DobotTask.P2_143:
        move_for_p2_143(client_feedback, client, topic, qos)
    else:
        logger.error(f"Wrong task passed in, task received: {task!r}")

    # close them here instead of in the run() function
    client_dashboard.close()
    client_feedback.close()

    client.loop_stop()
    client.disconnect()


def data_feedback(client_feedback: dobot_api_feedback):
    # The feedback information about port 30003 is displayed
    while True:
        time.sleep(0.05)
        all = client_feedback.socket_feedback.recv(10240)
        data = all[0:1440]
        a = np.frombuffer(data, dtype=MyType)
        if hex((a['test_value'][0])) == '0x123456789abcdef':
            print('robot_mode', a['robot_mode'])
            print('tool_vector_actual', np.around(
                a['tool_vector_actual'], decimals=4))
            print('q_actual', np.around(a['q_actual'], decimals=4))


def run(conf: MQTTConfig, task: DobotTask = DobotTask.Box):
    # Enable threads on ports 29999 and 30003
    client_dashboard = dobot_api_dashboard('192.168.5.1', 29999)
    client_feedback = dobot_api_feedback('192.168.5.1', 30003)

    p1 = Thread(
        target=move_and_publish_view,
        args=(client_dashboard, client_feedback, conf, task)
    )
    p1.start()

    # Not using all these for our vision inspection app
    # p2 = Thread(target=data_feedback, args=(client_feedback,))
    # p2.daemon = True
    # p2.start()
    # p1.join()
    # client_dashboard.close()
    # client_feedback.close()


if __name__ == '__main__':
    conf = MQTTConfig()
    task = DobotTask.Box
    run(conf, task)
