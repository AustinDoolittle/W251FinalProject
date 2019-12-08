import paho.mqtt.client as mqtt
MQTT_BROKER_IN='pose_broker_1'
MQTT_BROKER_OUT='50.22.176.123'
MQTT_TOPIC_IN='edge_capture'
MQTT_TOPIC_OUT='cloud_capture'


def on_message(client, userdata, message):
    #client_out = mqtt.Client()
    #client_out.connect(MQTT_BROKER_OUT, 1884, 60)
    client_out.publish(MQTT_TOPIC_OUT, payload=message.payload, qos=0)
    #client_out.disconnect()
    #print(message.payload)

client_out = mqtt.Client()
client_out.connect(MQTT_BROKER_OUT, 1884, 60)


client_in = mqtt.Client()
client_in.connect(MQTT_BROKER_IN,1883)
client_in.on_message = on_message
client_in.subscribe(MQTT_TOPIC_IN)
client_in.loop_forever()
