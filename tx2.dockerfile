FROM w251/keras:dev-tx2-4.2_b158-py3 

RUN apt-get update && \
    apt-get install -u -y  \        
        python3 \
        python3-pip

RUN pip3 install requests

COPY download.py /tmp/
COPY constants.py /tmp/

RUN python3 /tmp/download.py  --output-model-file /tmp/model.pb

FROM w251/keras:dev-tx2-4.2_b158-py3
ARG OPEN_CV_URL=http://169.44.201.108:7002/jetpacks/4.2.1

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

RUN pip3 install paho-mqtt

# create our workspace
RUN mkdir /app

# copy the model from the build step
COPY --from=0 /tmp/model.pb /app

# copy our source
COPY constants.py /app
COPY decode.py /app
COPY model.py /app
COPY find_person.py /app

# set the entrypoint
#ENTRYPOINT python3
ENTRYPOINT python3 /app/find_person.py --model-file /app/model.pb
