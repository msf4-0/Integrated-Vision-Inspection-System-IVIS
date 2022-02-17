from multiprocessing import Process
from time import sleep
from paho.mqtt.client import Client
from deployment.utils import (MQTTConfig)
import pandas as pd
from pathlib import Path
import os


def csv_label_checking(path: Path) -> Path:
    inspection_topic = MQTTConfig.topics.dobot_view
    inspection_qos = MQTTConfig.qos
    host = "localhost"
    client = Client("CSV")
    client.connect(host, port=1883)

    csv_label_fpath = path / "inspection_labels.csv"
    df = pd.read_csv(str(csv_label_fpath))
    data_dict = df.to_dict(orient='records')

    for row in data_dict:
        for key, labels in row.items():
            label_to_check = '{' + labels + '}'
            client.publish(inspection_topic, label_to_check, inspection_qos)
            sleep(1)


def run(arg):
    p1 = Process(target=csv_label_checking, args=arg)
    p1.start()

    return True, p1
