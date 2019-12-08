import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
MQTT_BROKER_IN='pose_broker_1'
MQTT_BROKER_OUT='50.22.176.123'
MQTT_TOPIC_IN='edge_capture'
MQTT_TOPIC_OUT='cloud_capture'


def on_message(client, userdata, message):
    r = publish.single(MQTT_TOPIC_OUT, payload=message.payload, qos=0, hostname=MQTT_BROKER_OUT,
         port= 1884)


client_in = mqtt.Client()
client_in.connect(MQTT_BROKER_IN,1883)
client_in.on_message = on_message
client_in.subscribe(MQTT_TOPIC_IN)
client_in.loop_forever()
