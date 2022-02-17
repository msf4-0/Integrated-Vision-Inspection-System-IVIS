from multiprocessing import Process
from time import sleep
from paho.mqtt.client import Client
from deployment.utils import (MQTTConfig)
import pandas as pd
from pathlib import Path
import os

# Reads the labels from the project's csv file and publish through MQTT
def csv_label_checking(path: Path) -> Path:
    # Set MQTT parameters
    inspection_topic = MQTTConfig.topics.dobot_view
    inspection_qos = MQTTConfig.qos
    host = "localhost"
    client = Client("CSV")
    client.connect(host, port=1883)

    # Get file path of the project's csv file 
    csv_label_fpath = path / "inspection_labels.csv"

    # Reads the labels and store in as an object 
    df = pd.read_csv(str(csv_label_fpath))
    data_dict = df.to_dict(orient='records')

    #iterates through all the labels and publish them to the topic
    for row in data_dict:
        for key, labels in row.items():
            label_to_check = '{' + labels + '}'
            client.publish(inspection_topic, label_to_check, inspection_qos)
            sleep(1)

#Runs the above fucntion as a process
def run(arg):
    p1 = Process(target=csv_label_checking, args=arg)
    p1.start()

    return True, p1
