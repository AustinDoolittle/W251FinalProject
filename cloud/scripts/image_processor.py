import paho.mqtt.client as mqtt
import datetime as dt
import os
import base64
import uuid
import json
import pymongo

import ibm_boto3
from ibm_botocore.client import Config, ClientError

client_uuid = str(uuid.uuid1())

MQTT_BROKER_IN = 'cloud_broker_1'
BUCKET_NAME = 'w251projectsds'
TOPIC_NAME = 'cloud_capture'

MONGO_INSTANCE = 'cloud_mongo_1'
MONGO_PORT = 27017
MONGO_DB = 'yoga_poser'
MONGO_COLLECTION = 'poses'

mongo_client = pymongo.MongoClient('mongodb://{}:{}'.format(MONGO_INSTANCE, MONGO_PORT))
mongo_db = mongo_client[MONGO_DB]
mongo_collection = mongo_db[MONGO_COLLECTION]

with open('/root/.bluemix/cos_credentials') as f:
    s3_credentials = json.loads(f.read())

COS_ENDPOINT = 'https://s3.us-south.cloud-object-storage.appdomain.cloud' 
COS_API_KEY_ID = s3_credentials['apikey']
COS_AUTH_ENDPOINT = "https://iam.cloud.ibm.com/identity/token"
COS_RESOURCE_CRN = s3_credentials['resource_instance_id']
COS_ENDPOINT = 'https://s3.private.us-south.cloud-object-storage.appdomain.cloud'
COS_BUCKET = 'yoga-poser'


cos = ibm_boto3.resource("s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_RESOURCE_CRN,
    ibm_auth_endpoint=COS_AUTH_ENDPOINT,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
    )


def on_message(client, userdata, message):
    msg = json.loads(message.payload.decode('utf-8'))
    client_uuid = msg['client']
    rn = dt.datetime.now()
    dt_str = rn.strftime("%Y%m%d_%H%M%f")
    save_path = '/data/images/{}/{}/{}/{}'.format(BUCKET_NAME, str(rn.year), str(rn.month), str(rn.day))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = 'image_{}_{}.jpg'.format(client_uuid, dt_str)
    full_name = '{}/{}'.format(save_path, file_name)
    #with open(full_name, 'wb') as f_output:
    #    f_output.write(base64.b64decode(msg.pop('image')))
    
    try:
        obj = cos.Object(COS_BUCKET, full_name)
        obj.put(Body = base64.b64decode(msg.pop('image')) )

    except Exception as e:
        print(e)
    
    msg['file_name'] = file_name
    msg['capture_dt'] = dt_str
    mongo_result = mongo_collection.insert_one(msg)
    
    return

client = mqtt.Client()
client.on_message = on_message
client.connect(MQTT_BROKER_IN, 1884, 60)
client.subscribe(TOPIC_NAME)
client.loop_forever()
