#!/usr/bin/env bash
set -e
cd "${0%/*}"/cloud
./docker_cloud.sh

cd ../edge
./docker_edge.sh
