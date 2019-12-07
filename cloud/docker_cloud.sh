#!/usr/bin/env bash
set -e

docker network create --driver bridge cloudnet

docker build -t cloud_broker -f docker/cloud_broker.dockerfile .
docker run -d --rm --name cloud_broker_1 --network cloudnet -p 1884:1884 cloud_broker

mkdir -p /data/images
docker build -t cloud_processor -f docker/cloud_processor.dockerfile .
docker run -d --rm --name cloud_processor_1 -v /data/images:/data/images --network cloudnet cloud_processor


mkdir -p /data/db

docker run -d --rm --name cloud_mongo -v /data/db:/data/db --network cloudnet -e MONGO_INITDB_ROOT_PASSWORD_FILE=credentials/.mongodb_secrets -p 27017:27017 mongo
