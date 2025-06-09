# mqtt_config.py

import json
from datetime import datetime

# MQTT 설정
BROKER_ADDRESS = "99.999.999.99" #INSERT YOUR BROKER ADDRESS
PORT = 9999 # INSERT YOUR MQTT PORT 
TOPIC = "/topic/KETI/csi" # INSERT YOUR MQTT TOPIC
MAC_ADDRESS = ["MAC_ADDRESS_1", "MAC_ADDRESS_2"] # MAC_ADRESS LIST


# 패킷 ID 관리
packet_id = -1

def get_next_packet_id():
    global packet_id
    packet_id += 1
    if packet_id > 50000:
        packet_id = 0
    return packet_id

def create_mqtt_message(loc=None, act=None,time=None):
    message = {
        "packet_id": get_next_packet_id(),
        "timestamp": time,
        "loc" : loc,
        "act" : act,
    }
    return json.dumps(message)
