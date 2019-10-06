FROM w251/keras:dev-tx2-4.2_b158-py3
ARG OPEN_CV_URL=http://169.44.201.108:7002/jetpacks/4.2.1
ARG MODEL_URL=https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite

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
        libv4l-dev \
        python \
        python3 \
        python-pip \
        python3-pip \
        curl 

# download and install opencv
RUN curl ${OPEN_CV_URL}/libopencv_3.3.1-2-g31ccdfe11_arm64.deb  -so /tmp/libopencv_3.3.1-2-g31ccdfe11_arm64.deb && \
    curl ${OPEN_CV_URL}/libopencv-dev_3.3.1-2-g31ccdfe11_arm64.deb -so /tmp/libopencv-dev_3.3.1-2-g31ccdfe11_arm64.deb && \
    curl ${OPEN_CV_URL}/libopencv-python_3.3.1-2-g31ccdfe11_arm64.deb -so /tmp/libopencv-python_3.3.1-2-g31ccdfe11_arm64.deb

RUN dpkg -i /tmp/*.deb

# clean up
RUN rm -rf /tmp/*

# create our workspace
RUN mkdir /app

# download the model
RUN curl -o /app/model.tflite ${MODEL_URL}

# copy our source
COPY find_person.py /app

# set the entrypoint
# ENTRYPOINT python3
ENTRYPOINT python3 /app/find_person.py
