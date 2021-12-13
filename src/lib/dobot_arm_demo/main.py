from typing import Dict, Tuple
from dobot_api import dobot_api_dashboard, dobot_api_feedback, MyType
from multiprocessing import Process
import numpy as np
import time

from streamlit import session_state

# view -> (label, required_number_of_the_label)
BOX_VIEW_LABELS: Dict[str, Tuple[str, int]] = {
    'top': ('hexagon', 3),
    'back': ('rectangle', 1),
    'right': ('circle', 2),
    'front': ('omron-sticker', 1),
    'left': ('triangle', 2),
}


def move(client_dashboard: dobot_api_dashboard, client_feedback: dobot_api_feedback):
    # Remove alarm
    client_dashboard.ClearError()
    time.sleep(0.5)
    # Description The upper function was enabled successfully
    client_dashboard.EnableRobot()
    time.sleep(0.5)

    # move to origin (top view)
    client_feedback.JointMovJ(
        (0.05), (-38.74), (-118.19), (157.46), (87.44), (0))
    time.sleep(5)

    # set check_labels to check 'top' view (check for this session_state in video loop)
    session_state.check_labels = 'top'
    time.sleep(1)
    session_state.check_labels = None

    # move to back side
    client_feedback.JointMovJ(
        (0.54), (-50.16), (-153.78), (114.97), (89.54), (-178))
    time.sleep(5)

    session_state.check_labels = 'back'
    time.sleep(1)
    session_state.check_labels = None

    # move to right side
    client_feedback.JointMovJ(
        (27.478), (-47.836), (-111.595), (70.479), (89.5876), (-63.62))
    time.sleep(5)

    session_state.check_labels = 'right'
    time.sleep(1)
    session_state.check_labels = None

    # move to front side
    client_feedback.JointMovJ(
        (0.56), (-76.07), (-37.9), (25.09), (90.11), (-2.21))
    time.sleep(5)

    session_state.check_labels = 'front'
    time.sleep(1)
    session_state.check_labels = None

    # move to left side
    client_feedback.JointMovJ(
        (-24.19), (-59.51), (-79), (49.49), (90.57), (62.96))
    time.sleep(5)

    session_state.check_labels = 'left'
    time.sleep(1)
    session_state.check_labels = None

    # move back to origin (top view)
    client_feedback.JointMovJ(
        (0.05), (-38.74), (-118.19), (157.46), (87.44), (0))
    time.sleep(5)
    session_state.check_labels = 'end'

    # close them here instead of in the run() function
    client_dashboard.close()
    client_feedback.close()


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


def run():
    # Enable threads on ports 29999 and 30003
    client_dashboard = dobot_api_dashboard('192.168.5.1', 29999)
    client_feedback = dobot_api_feedback('192.168.5.1', 30003)
    p1 = Process(target=move, args=(client_dashboard, client_feedback))
    p1.daemon = True
    p1.start()

    # Not using all these for our vision inspection app
    # p2 = Process(target=data_feedback, args=(client_feedback,))
    # p2.daemon = True
    # p2.start()
    # p1.join()
    # client_dashboard.close()
    # client_feedback.close()


if __name__ == '__main__':
    run()
