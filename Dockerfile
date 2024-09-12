# Use the official NVIDIA TensorFlow image as a base
FROM nvcr.io/nvidia/tensorflow:24.08-tf2-py3

# Set environment variables for CUDA
ENV CUDA_VISIBLE_DEVICES=0

# Install any additional dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy your project files into the container
COPY . /workspace
WORKDIR /workspace

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Set the entrypoint for the container
ENTRYPOINT ["python3", "main.py"]