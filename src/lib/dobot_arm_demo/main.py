from typing import Dict, List, Tuple
from pathlib import Path
import sys
from enum import IntEnum

from paho.mqtt.client import Client

from dobot_api import dobot_api_dashboard, dobot_api_feedback, MyType
from multiprocessing import Process
from threading import Thread
import numpy as np
import time


SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

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
    'front left': [('left weld', 1)],
    'front right': [('right weld', 1)],
}


def get_and_start_client(conf: MQTTConfig, topic: str, qos: int) -> Client:
    client = get_mqtt_client()
    client.connect(conf.broker, port=conf.port)
    client.subscribe(topic, qos)
    client.loop_start()
    return client


def stop_client(client: Client):
    client.loop_stop()
    client.disconnect()


def move(client_dashboard: dobot_api_dashboard, client_feedback: dobot_api_feedback,
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
        move_for_box(client_feedback, client)
    elif task == DobotTask.P2_143:
        move_for_p2_143(client_feedback, client)

    # close them here instead of in the run() function
    client_dashboard.close()
    client_feedback.close()

    client.loop_stop()
    client.disconnect()


def move_for_box(client_feedback: dobot_api_feedback, client: Client):
    # move to origin (top view)
    client_feedback.JointMovJ(
        (0.05), (-38.74), (-118.19), (157.46), (87.44), (0))
    time.sleep(2)

    # send the current view as the payload to our vision inspection app
    client.publish(topic, 'top', qos)
    time.sleep(1)
    client.publish(topic, None, qos)

    # move to back side
    client_feedback.JointMovJ(
        (0.54), (-50.16), (-153.78), (114.97), (89.54), (-178))
    time.sleep(8)

    client.publish(topic, 'back', qos)
    time.sleep(2)
    client.publish(topic, None, qos)

    # move to right side
    client_feedback.JointMovJ(
        (27.478), (-47.836), (-111.595), (70.479), (89.5876), (-63.62))
    time.sleep(5)

    client.publish(topic, 'right', qos)
    time.sleep(2)
    client.publish(topic, None, qos)

    # move to front side
    client_feedback.JointMovJ(
        (0.56), (-76.07), (-37.9), (25.09), (90.11), (-2.21))
    time.sleep(5)

    client.publish(topic, 'front', qos)
    time.sleep(2)
    client.publish(topic, None, qos)

    # move to left side
    client_feedback.JointMovJ(
        (-24.19), (-59.51), (-79), (49.49), (90.57), (62.96))
    time.sleep(4)

    client.publish(topic, 'left', qos)
    time.sleep(2)
    client.publish(topic, None, qos)

    # move back to origin (top view)
    client_feedback.JointMovJ(
        (0.05), (-38.74), (-118.19), (157.46), (87.44), (0))
    time.sleep(5)
    client.publish(topic, 'end', qos)


def move_for_p2_143(client_feedback: dobot_api_feedback, client: Client):
    # move to origin (top view)
    client_feedback.JointMovJ(
        (0.05), (-38.74), (-118.19), (157.46), (87.44), (0))
    time.sleep(2)

    # send the current view as the payload to our vision inspection app
    client.publish(topic, 'top', qos)
    time.sleep(1)
    client.publish(topic, None, qos)

    # move to back side
    client_feedback.JointMovJ(
        (0.54), (-50.16), (-153.78), (114.97), (89.54), (-178))
    time.sleep(5)

    client.publish(topic, 'top left', qos)
    time.sleep(2)
    client.publish(topic, None, qos)

    # move to right side
    client_feedback.JointMovJ(
        (27.478), (-47.836), (-111.595), (70.479), (89.5876), (-63.62))
    time.sleep(5)

    client.publish(topic, 'top right', qos)
    time.sleep(2)
    client.publish(topic, None, qos)

    # move to front side
    client_feedback.JointMovJ(
        (0.56), (-76.07), (-37.9), (25.09), (90.11), (-2.21))
    time.sleep(5)

    client.publish(topic, 'front left', qos)
    time.sleep(2)
    client.publish(topic, None, qos)

    # move to left side
    client_feedback.JointMovJ(
        (-24.19), (-59.51), (-79), (49.49), (90.57), (62.96))
    time.sleep(5)

    client.publish(topic, 'front right', qos)
    time.sleep(2)
    client.publish(topic, None, qos)

    # move back to origin (top view)
    client_feedback.JointMovJ(
        (0.05), (-38.74), (-118.19), (157.46), (87.44), (0))
    time.sleep(5)
    client.publish(topic, 'end', qos)


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

    if task == DobotTask.Box:
        move_func = move_for_box
    elif task == DobotTask.P2_143:
        move_func = move_for_p2_143

    p1 = Thread(target=move_func, args=(
        client_dashboard, client_feedback,
        conf, topic, qos))
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
    topic = conf.topics.dobot_view
    qos = 1
    run(conf, topic, qos)
