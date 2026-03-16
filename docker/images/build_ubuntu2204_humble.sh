#! /bin/bash

docker build --build-arg USERNAME=docker --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --network=host -f Dockerfile.ubuntu2204_humble -t gidobot:knfu_slam_2204_humble ../