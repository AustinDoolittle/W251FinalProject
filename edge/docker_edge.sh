#!/usr/bin/env bash
set -e

xhost +
sudo nvpmodel -m 0
sudo jetson_clocks

docker network create --driver bridge localnet

docker build -t pose_broker -f docker/pose_broker.dockerfile .
docker run -d --rm --name pose_broker_1 --network localnet -p 1883:1883 pose_broker 

docker build -t pose_forwarder -f docker/pose_forwarder.dockerfile .
docker run -d --rm --name pose_forwarder_1 --network localnet pose_forwarder

docker build -t poser -f docker/tx2.dockerfile .
docker run \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp:/tmp \
    -v /dev/video1:/dev/video0 \
    -v /data:/data \
    --rm \
    --network localnet \
    --name pose_capture_1 \
    poser \
    --max-poses 1 \
    --pose-threshold 0.25 \
    --part-threshold 0.25 

