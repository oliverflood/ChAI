FROM ubuntu:22.04

COPY deb /

RUN apt update

RUN apt upgrade -y

RUN apt install -y ./chapel-2.3.0-1.ubuntu22.arm64.deb || apt install -y ./chapel-2.3.0-1.ubuntu22.amd64.deb

RUN apt install -y python3-pip

# docker build -t chapel-deb .
# docker container run -it chapel-deb bash