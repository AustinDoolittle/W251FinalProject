#!/usr/bin/env bash
set -e

xhost +
docker build -t poser -f poser.dockerfile .
docker run \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp:/tmp \
    -v /dev/video1:/dev/video0 \
    poser \
    --max-poses 1 \
    --pose-threshold 0.25 \
    --part-threshold 0.25 \

