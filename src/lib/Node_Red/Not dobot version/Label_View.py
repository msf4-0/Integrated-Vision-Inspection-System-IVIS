from threading import Thread
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

client = mqtt.Client("VisionSysApp")
client.connect(host, port=1883)
client.loop_start()

# Subscribe to all topics
for topic in TOPICS:
    client.subscribe(topic)

client.on_message = main_cb


def main():
    while True:
        # Get data from CSV file
        if recv_topic == "topic/runCSV":
            print(recv_msg)
            labels = recv_msg['labels']
            view = recv_msg['view']
            print(labels)
            print(view)
            time.sleep(1)
            break

        # save filename
        if recv_topic == "topic/filename":
            filename = recv_msg
            filepath = "Desktop"
            fullpath = filepath + filename
            recvfile = 1
        # export data to file
        if recv_topic == "topic/recv_data":
            # access the Dictionary from the parsed JSON
            print(recv_msg)
            labels = recv_msg[0]['labels']
            view = recv_msg[0]['view']

            # save to CSV file
            df = pd.DataFrame([recv_msg]).T
            df.to_csv(fullpath, index=False)

        # receive data from csv file and send to node red
        if recv_topic == "topic/CoreFunc" and recv_msg == "GetData" and recvfile == 1:
            df = pd.read_csv(fullpath)
            # minus 2 because it will have data + 2 more lines
            lines = len(df)-2

            print(lines)
            print(row)
            data_dict = "Random String"
            while row <= lines:
                time.sleep(1)
                print(row)

                data_dict = df.to_dict(orient='records')[row]
                print(f"{data_dict}")
                send = str(data_dict)
                client.publish("topic/recv_csv", send)
                row = row + 1
            data_dict = "Random String"
            recvfile = 0


# Enable threads on ports 29999 and 30003
if __name__ == '__main__':
    p1 = Thread(target=main)
    p1.start()
