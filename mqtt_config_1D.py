import json
from datetime import datetime

# Configure MQTT Information
BROKER_ADDRESS = "10.999.99.99" #INSERT YOUR BROKER ADDRESS
PORT = 9999 # INSERT YOUR MQTT PORT 
TOPIC = "/topic/quber/csi" # INSERT YOUR MQTT TOPIC
MAC_ADDRESS = ["99:99:99:99:99:99"]

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
