#!/usr/bin/env bash

docker build --tag 'chapel_pytorch_test' .
docker run -a STDOUT -a STDERR 'chapel_pytorch_test'

# terminal cmd: sh build-run.sh && docker run -it 'chapel_pytorch_test'
#     when testing and output of Dockerfile could be unreliable. Having
#     this allows you to see the environment produced by the Dockerfile!