#!/usr/bin/env bash
set -e

xhost +
docker build -t poser -f poser.dockerfile .
docker run \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    --device /dev/video1:/dev/video0 \
    --runtime nvidia \
    poser
