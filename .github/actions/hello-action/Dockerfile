# Container image that runs your code
# FROM ubuntu:latest
# FROM pytorch/pytorch:latest

FROM chapel/chapel:latest as chapel

RUN echo $(which chpl)

FROM pytorch/pytorch:latest

COPY --from=chapel /opt/chapel/bin/chpl /opt/chapel/bin/chpl

ENV PATH="/opt/chapel/bin/chpl:${PATH}"

# ADD chapel/chapel /chapel

WORKDIR /ChAI

COPY . .

COPY ../../../ ./


# Copies your code file from your action repository to the filesystem path `/` of the container
# COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

RUN echo $(which chpl || echo "bad chapel. ")

# Code file to execute when the docker container starts up (`entrypoint.sh`)
ENTRYPOINT ["/entrypoint.sh"]