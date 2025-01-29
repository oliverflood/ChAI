FROM chapel/chapel:latest

# Install Python 3, pip, NumPy, Torch
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && pip3 install --no-cache-dir numpy torch \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

