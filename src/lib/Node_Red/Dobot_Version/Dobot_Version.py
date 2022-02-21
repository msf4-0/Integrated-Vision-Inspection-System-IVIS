from dobot_api import dobot_api_dashboard, dobot_api_feedback, MyType
from multiprocessing import Process
import numpy as np
import time
import json
import paho.mqtt.client as mqtt
import pandas as pd

# initialize for the topic on which the msg is received
recv_topic = ""
# initialize a recv msg to store the payload received from MQTT
recv_msg = ""

# define all topics
TOPICS = [
    #EStop, En/Dis, ClearErr
    "topic/CoreFunc",
    # Export teach points data
    "topic/recv_data",
    # filename of csv file
    "topic/filename",
    # runCSV
    "topic/runCSV"
]


def main_cb(client, userdata, msg):
    # use `global` to change a variable outside of the callback function
    global recv_topic, recv_msg
    recv_topic = msg.topic
    print('In callback, topic:', recv_topic)
    if recv_topic == "topic/recv_data" or recv_topic == "topic/runCSV":
        recv_msg = json.loads(msg.payload)
        print('In callback, msg:', recv_msg)
    else:
        recv_msg = msg.payload.decode('utf-8')
        print('In callback, msg:', recv_msg)


# localhost / 127.0.0.1
host = "localhost"

client = mqtt.Client("DobotCR5")
client.connect(host, port=1883)
client.loop_start()

# Subscribe to all topics
for topic in TOPICS:
    client.subscribe(topic)

client.on_message = main_cb


# def debug_run():
#     def debug_publish():
#         while True:
#             for topic in TOPICS:
#                 client.publish(topic, 'dummy message', qos=1)
#                 time.sleep(0.5)

#                 # the recv_topic & recv_msg can be accessed in main function
#                 # then you can do your stuff with them here
#                 print(f"In main function, topic: {recv_topic}")
#                 print(f"In main function, msg: {recv_msg}\n")

#                 if recv_topic == 'topic/Terminate':
#                     print("TERMINATING")
#                     client.loop_stop()
#                     client.disconnect()
#                     exit()

#     p1 = Thread(target=debug_publish)
#     p1.start()

# Main function where all robot code logic exist
size = 6
Angle = [0] * size
Pose = [0] * size
file = 0


def main(client_dashboard, client_feedback, Angle, Pose):
    # Remove alarm
    client_dashboard.ClearError()
    time.sleep(0.5)
    # Description The upper function was enabled successfully
    client_dashboard.EnableRobot()
    time.sleep(0.5)
    while True:
        # Continuously get angle and position data
        Angle = client_dashboard.GetAngle()
        client.publish("topic/Angle", Angle)
        Pose = client_dashboard.GetPose()
        client.publish("topic/Pose", Pose)
        time.sleep(0.25)

        # Move robot with csv file
        if recv_topic == "topic/runCSV":
            print(recv_msg)
            coord1 = float(recv_msg['J1'])
            coord2 = float(recv_msg['J2'])
            coord3 = float(recv_msg['J3'])
            coord4 = float(recv_msg['J4'])
            coord5 = float(recv_msg['J5'])
            coord6 = float(recv_msg['J6'])
            labels = recv_msg['labels']
            view = recv_msg['view']
            Angle = client_dashboard.GetAngle()
            client_feedback.JointMovJ(float(coord1), float(coord2), float(
                coord3), float(coord4), float(coord5), float(coord6))
            time.sleep(2)

        # save filename
        if recv_topic == "topic/filename":
            filename = recv_msg
            filepath = "Desktop"
            fullpath = filepath + filename
            file = 1
        # export data to file
        if recv_topic == "topic/recv_data":
            # access the Dictionary from the parsed JSON
            coord1 = recv_msg[0]['J1']
            coord2 = recv_msg[0]['J2']
            coord3 = recv_msg[0]['J3']
            coord4 = recv_msg[0]['J4']
            coord5 = recv_msg[0]['J5']
            coord6 = recv_msg[0]['J6']
            labels = recv_msg[0]['labels']
            view = recv_msg[0]['view']

            # save to CSV file
            df = pd.DataFrame([recv_msg]).T
            df.to_csv(fullpath, index=False)

        if recv_topic == "topic/CoreFunc" and recv_msg == "GetData" and file == 1:
            df = pd.read_csv(fullpath)
            # minus 2 because it will have data + 2 more lines
            lines = len(df) - 2
            #df = pd.read_csv('OneDrive/Desktop/Internship/test2.csv')
            row = 0
            data_dict = "Random String"
            while row <= lines:
                time.sleep(1)
                data_dict = df.to_dict(orient='records')[row]
                print(f"{data_dict}")
                send = str(data_dict)
                client.publish("topic/recv_joint", send)
                row = row + 1
            data_dict = "Random String"
            file = 0


# Enable threads on ports 29999 and 30003
def run():
    client_dashboard = dobot_api_dashboard('192.168.5.1', 29999)
    client_feedback = dobot_api_feedback('192.168.5.1', 30003)
    p1 = Process(target=main, args=(
        client_dashboard, client_feedback, Angle, Pose))
    p1.start()
    return True, p1
