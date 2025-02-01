# Container image that runs your code
# FROM ubuntu:latest
# FROM pytorch/pytorch:latest

FROM chapel/chapel
# as chapel

# Install pytorch
RUN pip3 install torch numpy torch

# PLEASE SEE {ChAI}/docker/chapel-pytorch-test/.!




#
#
#
#
#
#  See docker/chapel-pytorch-test and examine it closely!
#
#
#
#
#
























RUN echo $(which python3)
RUN echo $(which chpl)

RUN echo $(ls -la)

WORKDIR /ChAI

RUN echo $(ls -la)

COPY . .

RUN echo $(ls -la)

COPY entrypoint.sh /entrypoint.sh
COPY ../../../ ./

RUN echo $(ls -la)


# Copies your code file from your action repository to the filesystem path `/` of the container
# COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

RUN echo $(which chpl || echo "bad chapel. ")

# Code file to execute when the docker container starts up (`entrypoint.sh`)
ENTRYPOINT ["/entrypoint.sh"]