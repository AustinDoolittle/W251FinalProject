import paho.mqtt.client as mqtt
import datetime as dt
import os
import base64
import uuid
import json

client_uuid = str(uuid.uuid1())

MQTT_BROKER_IN = 'cloud_broker_1'
BUCKET_NAME = 'w251projectsds'
TOPIC_NAME = 'cloud_capture'

def on_message(client, userdata, message):
    msg = json.loads(message.payload.decode('utf-8'))
    client_uuid = msg['client']
    rn = dt.datetime.now()
    dt_str = rn.strftime("%Y%m%d_%H%M%f")
    save_path = '/tmp/{}/{}/{}/{}/'.format(BUCKET_NAME, str(rn.year), str(rn.month), str(rn.day))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fn = '{}/image_{}_{}.jpg'.format(save_path, client_uuid, dt_str)
    with open(fn, 'wb') as f_output:
        f_output.write(base64.b64decode(msg['image']))
    return

client = mqtt.Client()
client.on_message = on_message
client.connect(MQTT_BROKER_IN, 1884, 60)
client.subscribe(TOPIC_NAME)
client.loop_forever()
