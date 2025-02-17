FROM ubuntu:22.04


RUN apt update && apt upgrade -y > /dev/null


COPY deb /

RUN (apt install -y ./chapel-2.3.0-1.ubuntu22.arm64.deb > /dev/null) || (apt install -y ./chapel-2.3.0-1.ubuntu22.amd64.deb > /dev/null)

RUN apt install -y python3-pip > /dev/null

RUN pip3 install numpy torch torchvision > /dev/null


COPY lib /lib
COPY examples /examples
COPY test /test

# Run tests
RUN cd test/correspondence && python3 correspondence.py
RUN cd test/correspondence && python3 correspondence.py --print-compiler-errors


# Build instructions
# force (last resort): docker system prune --all --force
# docker builder prune --all
# docker build --no-cache -t chapel-deb .
# docker build -t chapel-deb .
# docker container run -it chapel-deb bash