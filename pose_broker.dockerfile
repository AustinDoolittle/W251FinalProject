FROM alpine

RUN apk update && apk add mosquitto

ENTRYPOINT mosquitto
