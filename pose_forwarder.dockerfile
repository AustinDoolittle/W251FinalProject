FROM alpine

RUN apk update && apk add mosquitto-clients
RUN apk add --update python3 py3-pip 

RUN pip3 install paho-mqtt

RUN mkdir /app
COPY subscribe_and_forward.py /app

ENTRYPOINT python3 /app/subscribe_and_forward.py
