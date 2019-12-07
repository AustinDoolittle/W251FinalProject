FROM ubuntu

RUN apt-get update
RUN apt-get install -y automake autotools-dev g++ git libcurl4-openssl-dev libfuse-dev libssl-dev libxml2-dev make pkg-config python3 python3-pip vim

RUN pip3 install paho-mqtt

RUN mkdir /app

COPY scripts/image_processor.py /app

ENTRYPOINT python3 /app/image_processor.py
