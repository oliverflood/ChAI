FROM ubuntu:22.04


RUN apt update

RUN apt upgrade -y


COPY deb /

RUN apt install -y ./chapel-2.3.0-1.ubuntu22.arm64.deb || apt install -y ./chapel-2.3.0-1.ubuntu22.amd64.deb

RUN apt install -y python3-pip

RUN pip3 install numpy torch torchvision


COPY lib /lib
COPY examples /examples
COPY test /test

# Run tests
RUN cd test/correspondence && python3 correspondence.py


# Build instructions
# force (last resort): docker system prune --all --force
# docker builder prune --all
# docker build --no-cache -t chapel-deb .
# docker build -t chapel-deb .
# docker container run -it chapel-deb bash