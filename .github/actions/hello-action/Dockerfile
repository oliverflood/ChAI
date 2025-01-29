# Container image that runs your code
# FROM ubuntu:latest
# FROM pytorch/pytorch:latest
FROM pytorch/pytorch:latest AS builder

FROM chapel/chapel

WORKDIR /ChAI

COPY . .

COPY ../../../ ./

COPY chai/chai ./
# ADD chai/chai

# Copies your code file from your action repository to the filesystem path `/` of the container
# COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

# Code file to execute when the docker container starts up (`entrypoint.sh`)
ENTRYPOINT ["/entrypoint.sh"]