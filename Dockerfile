# This image includes CUDA v11.2 and cuDNN
FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04 AS builder-image

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# avoid stuck build due to user prompt
ARG DEBIAN_FRONTEND=noninteractive

# install libpq-dev for PostgreSQL dependencies to install psycopg2 library for Python
# and also protobuf for TFOD installation
# then install python related stuff
RUN apt update && apt install --no-install-recommends -y git make libpq-dev protobuf-compiler \
    python3.8 python3.8-dev python3.8-venv python3-pip python3-wheel build-essential && \
    apt clean && rm -rf /var/lib/apt/lists/*

# create and activate virtual environment
# using final folder name to avoid path issues with packages
# NOTE: must activate here to be able to run 'python' commands afterwards
RUN python3.8 -m venv /home/venv
ENV PATH="/home/venv/bin:$PATH"

# install requirements
COPY requirements_no_hash.txt .
RUN pip install --no-cache-dir wheel
RUN pip install --no-cache-dir -r requirements_no_hash.txt

# COCO API installation, note that paths are based on path_desc.py
# these are same with the tfod_installation.py script
RUN mkdir /home/TFOD && git clone https://github.com/tensorflow/models /home/TFOD/models && \
    git clone https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && make && cp -r pycocotools /home/TFOD/models/research && \
    # need to upgrade pip to be able to use the '--use-feature' option later to install TFOD
    pip install pycocotools==2.0.2 && \
    pip install --upgrade pip

# TFOD installation
# these are same with the tfod_installation.py script
RUN cd /home/TFOD/models/research && \
    protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf2/setup.py setup.py && \
    pip install --use-feature=2020-resolver .

# main builder image, to save space from unnecessary files
FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04 AS runner-image

ENV PYTHONDONTWRITEBYTECODE=1

# avoid stuck for opencv installation user prompt
ENV TERM=xterm
ENV DEBIAN_FRONTEND=noninteractive

# psmisc is used to run 'killall' command to kill any existing tensorboard process
# python3-icu is required for the Python 'natsort' library to sort files like file browser
# libpq-dev is required for psycopg2 library
# ffmpeg for more video codecs to record videos
# libsm6 and the rest are required for OpenCV library
RUN apt-get update && apt-get install --no-install-recommends -y python3.8 python3-venv \
    psmisc python3-icu libpq-dev ffmpeg libsm6 libxext6 libxrender-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# copy Python virtual environment
COPY --from=builder-image /home/venv /home/venv

RUN mkdir -p /home/code/TFOD/models/research/object_detection
# copy TFOD stuff to the desired path of "TFOD_DIR" as defined in path_desc.py
COPY --from=builder-image /home/TFOD/models/research/object_detection \
    /home/code/src/lib/TFOD/models/research/object_detection
COPY --from=builder-image /home/TFOD/models/research/pycocotools \
    /home/code/src/lib/TFOD/models/research/pycocotools
RUN rm -rf /home/code/TFOD

WORKDIR /home/code
COPY . .

# expose our Streamlit port as defined in .streamlit/config.toml
EXPOSE 8502
# expose for TensorBoard, refer run_tensorboard() function
EXPOSE 6007

# activate virtual environment
ENV VIRTUAL_ENV=/home/venv
ENV PATH="/home/venv/bin:$PATH"

# environment variables for Streamlit config for deployment
# https://stackoverflow.com/questions/69352179/package-streamlit-app-and-run-executable-on-windows/69621578#69621578
# must set server to headless to avoid opening Streamlit directly on browser
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_GLOBAL_DEPLOYMENTMODE=false

# for database_setup to create database directly with database_direct_setup()
ENV DOCKERCOMPOSE=1

CMD ["streamlit", "run", "src/app.py"]

