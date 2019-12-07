FROM alpine

RUN apk update && apk add mosquitto

ENTRYPOINT mosquitto -p 1884
