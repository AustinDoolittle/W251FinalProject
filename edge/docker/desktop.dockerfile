FROM tensorflow/tensorflow:1.13.1-gpu-py3

# install OpenCV and TF deps
RUN apt-get update && \
    apt-get install -u -y  \
        libtbb-dev \
        libavcodec-dev \
        libavformat-dev \
        libhdf5-dev \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libgtk2.0-dev \
        libswscale-dev \
        libcanberra-gtk-module \
        libcanberra-gtk3-module \
        libv4l-dev \
        curl 

RUN pip install \
    opencv-python \
    keras

# create our workspace
RUN mkdir /app

# copy our source
COPY scripts/train_mlp.py /app

# set the entrypoint
# ENTRYPOINT python3
ENTRYPOINT python3 /app/train_classifier.py
