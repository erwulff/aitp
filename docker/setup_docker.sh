#!/bin/sh
docker build -f docker/Dockerfile --build-arg home=$HOME -t aitp .