FROM alpine

RUN apk update && apk add mosquitto-clients
RUN apk add --update python py-pip

RUN pip install paho-mqtt

RUN mkdir /app
COPY subscribe_and_forward.py /app

ENTRYPOINT python /app/subscribe_and_forward.py
