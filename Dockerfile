FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3.9 \
    python3-pip \
    ffmpeg \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy project files for installation
COPY thesis_sn-gamestate /app/thesis_sn-gamestate
COPY thesis_tracklab /app/thesis_tracklab

# Install all Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN cd /app/thesis_sn-gamestate && pip3 install -e .
RUN cd /app/thesis_tracklab && pip3 install -e . && pip3 install -r requirements.txt
RUN pip3 install openmim && mim install mmcv==2.0.1
RUN pip3 install "pytorch-lightning<2.0.0"

# Expose the port the API will run on
EXPOSE 8000

RUN export HYDRA_FULL_ERROR=1

# Set default command to run the FastAPI server
CMD ["python3", "-c", "import uvicorn; uvicorn.run('thesis_tracklab.api:app', host='0.0.0.0', port=8000, log_level='info')"] 
