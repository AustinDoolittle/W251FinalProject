FROM ubuntu

RUN apt-get update
RUN apt-get install -y automake autotools-dev g++ git libcurl4-openssl-dev libfuse-dev libssl-dev libxml2-dev make pkg-config python3 python3-pip vim

RUN pip3 install paho-mqtt pymongo ibm-cos-sdk==2.0.1

RUN mkdir /app
RUN mkdir $HOME/.bluemix

COPY scripts/image_processor.py /app
COPY credentials/.bucket_credentials /root/.bluemix/cos_credentials
ENTRYPOINT python3 /app/image_processor.py
