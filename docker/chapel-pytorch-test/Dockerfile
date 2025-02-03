FROM chapel/chapel

# Install pytorch
RUN pip3 install torch

# RUN echo "Hello!"
# RUN echo $(which chpl)

# ADD ./script.sh ./script.sh

WORKDIR /app

COPY ./script.sh script.sh
COPY . ../../.
RUN sh script.sh

# CMD ["sh","script.sh"] # Turn off when testing.