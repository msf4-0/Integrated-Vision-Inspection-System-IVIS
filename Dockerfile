# For more information, please refer to https://aka.ms/vscode-docker-python
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
RUN apt update && apt install -y git make libpq-dev protobuf-compiler && \
    apt install --no-install-recommends -y python3.8 python3.8-dev python3.8-venv python3-pip python3-wheel build-essential && \
    apt clean && rm -rf /var/lib/apt/lists/*

# create and activate virtual environment
# using final folder name to avoid path issues with packages
# NOTE: must activate here to be able to run 'python' commands afterwards
RUN python3.8 -m venv /home/myuser/venv
ENV PATH="/home/myuser/venv/bin:$PATH"

# install requirements
COPY requirements_no_hash.txt .
RUN pip install --no-cache-dir wheel
RUN pip install --no-cache-dir -r requirements_no_hash.txt
# RUN pip uninstall -y opencv-python && \
#     pip install opencv-python-headless

# COCO API installation, note that paths are based on path_desc.py
# these are same with the tfod_installation.py script
RUN mkdir /home/myuser/TFOD && git clone https://github.com/tensorflow/models /home/myuser/TFOD/models && \
    git clone https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && make && cp -r pycocotools /home/myuser/TFOD/models/research
# need to upgrade pip to be able to use the '--use-feature' option later to install TFOD
RUN pip install pycocotools==2.0.2 && \
    pip install --upgrade pip

# TFOD installation
# these are same with the tfod_installation.py script
RUN cd /home/myuser/TFOD/models/research && \
    protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf2/setup.py setup.py && \
    pip install --use-feature=2020-resolver .

# main builder image, to save space from unnecessary files
FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04 AS runner-image
# python3-icu is required for the Python 'natsort' library to sort files like file browser
# libpq-dev is required for psycopg2 library
# libgl1-mesa-glx and the rest are required for OpenCV library
RUN apt update && apt install --no-install-recommends -y python3.8 python3-venv
# Don't need all these, only need libgl1-mesa-glx and libglib2.0-0
# apt install -y python3-icu libpq-dev libgl1-mesa-glx ffmpeg libglib2.0-0 libsm6 libxrender1 libxext6 && \
RUN apt install -y python3-icu libpq-dev && \
    apt install -y libgl1-mesa-glx libglib2.0-0 && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home myuser
COPY --from=builder-image /home/myuser/venv /home/myuser/venv

# create the app's data directory and add read/write privilege for our user
RUN mkdir -p /home/myuser/.local/share/integrated-vision-inspection-system && \
    chown myuser /home/myuser/.local/share/integrated-vision-inspection-system

USER myuser
RUN mkdir -p /home/myuser/code/TFOD/models/research/object_detection
# copy TFOD stuff to the desired path of "TFOD_DIR" as defined in path_desc.py
COPY --from=builder-image /home/myuser/TFOD/models/research/object_detection \
    /home/myuser/code/src/lib/TFOD/models/research/object_detection
COPY --from=builder-image /home/myuser/TFOD/models/research/pycocotools \
    /home/myuser/code/src/lib/TFOD/models/research/pycocotools
RUN rm -rf /home/myuser/code/TFOD
WORKDIR /home/myuser/code
COPY . .

# expose our Streamlit port as defined in .streamlit/config.toml
EXPOSE 8502

# activate virtual environment
ENV VIRTUAL_ENV=/home/myuser/venv
ENV PATH="/home/myuser/venv/bin:$PATH"

# environment variables for Streamlit config for deployment
# https://stackoverflow.com/questions/69352179/package-streamlit-app-and-run-executable-on-windows/69621578#69621578
# must set server to headless to avoid opening Streamlit directly on browser
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_GLOBAL_DEPLOYMENTMODE=false

# for database_setup to create database directly with database_direct_setup()
ENV DOCKERCOMPOSE=1

# # make sure the container always starts with this ENTRYPOINT
# ENTRYPOINT ["streamlit", "run"]

# # this CMD can be overwritten to run Streamlit on another file
# CMD ["src/app.py"]
CMD ["streamlit", "run", "src/app.py"]

# FOR DEBUGGING GPU support
# CMD ["bash"]
