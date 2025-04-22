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

WORKDIR /app

# Copy only pyproject.toml and poetry.lock files first
COPY thesis_sn-gamestate/pyproject.toml /app/thesis_sn-gamestate/pyproject.toml
COPY thesis_sn-gamestate/plugins /app/thesis_sn-gamestate/plugins
RUN mkdir -p /app/thesis_sn-gamestate/sn_gamestate
RUN touch /app/thesis_sn-gamestate/sn_gamestate/__init__.py

COPY thesis_tracklab/requirements.txt /app/thesis_tracklab/requirements.txt
COPY thesis_tracklab/pyproject.toml /app/thesis_tracklab/pyproject.toml
COPY thesis_tracklab/plugins /app/thesis_tracklab/plugins
RUN mkdir -p /app/thesis_tracklab/tracklab
RUN touch /app/thesis_tracklab/tracklab/__init__.py

# Install Python dependencies and packages in editable mode
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip3 install -r /app/thesis_tracklab/requirements.txt
RUN cd /app/thesis_sn-gamestate && pip3 install -e .
RUN cd /app/thesis_tracklab && pip3 install -e .
RUN pip3 install openmim && mim install mmcv==2.0.1
RUN pip3 install "pytorch-lightning<2.0.0"

# Copy the remaining code after all installations
COPY thesis_sn-gamestate /app/thesis_sn-gamestate
COPY thesis_tracklab /app/thesis_tracklab

EXPOSE 8000

# CMD ["python3", "-m", "uvicorn", "tracklab.api:app", "--host", "0.0.0.0", "--port", "8000"] 
CMD ["python3", "-m", "tracklab.main", "-cn", "soccernet"]