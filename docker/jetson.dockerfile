FROM nvidia/cuda:11.4.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG OPENCV_VERSION=4.5.3

RUN apt-get update && apt-get upgrade -y &&\
    # Install build tools, build dependencies and python
    apt-get install -y \
	python3-pip \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libxine2-dev \
        libglew-dev \
        libtiff5-dev \
        zlib1g-dev \
        libjpeg-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libpostproc-dev \
        libswscale-dev \
        libeigen3-dev \
        libtbb-dev \
        libgtk2.0-dev \
        pkg-config \
        ## Python
        python3-dev \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# OpenCV
RUN cd /opt/ &&\
    # Download and unzip OpenCV and opencv_contrib and delte zip files
    wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip &&\
    unzip $OPENCV_VERSION.zip &&\
    rm $OPENCV_VERSION.zip &&\
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip &&\
    unzip ${OPENCV_VERSION}.zip &&\
    rm ${OPENCV_VERSION}.zip &&\
    # Create build folder and switch to it
    mkdir /opt/opencv-${OPENCV_VERSION}/build && cd /opt/opencv-${OPENCV_VERSION}/build &&\
    # Cmake configure
    cmake \
        -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
        -DWITH_CUDA=ON \
        -DCUDA_ARCH_BIN=7.5,8.0,8.6 \
        -DCMAKE_BUILD_TYPE=RELEASE \
        # Install path will be /usr/local/lib (lib is implicit)
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        .. &&\
    # Make
    make -j"$(nproc)" && \
    # Install to /usr/local/lib
    make install && \
    ldconfig &&\
    # Remove OpenCV sources and build folder
    rm -rf /opt/opencv-${OPENCV_VERSION} && rm -rf /opt/opencv_contrib-${OPENCV_VERSION}

# Boost with Python 3
RUN cd /tmp && wget -c 'https://sourceforge.net/projects/boost/files/boost/1.73.0/boost_1_73_0.tar.bz2'\
    && tar -xvf boost_1_73_0.tar.bz2 \
    && cd boost_1_73_0 \
    && bash bootstrap.sh --with-python=python3 \
    && ./b2 && ./b2 install \
    && cd / && rm -rf /tmp/boost_1_73_0*
    
# Eigen3
RUN cd /tmp && wget -c "https://gitlab.com/libeigen/eigen/-/archive/3.3.2/eigen-3.3.2.tar.bz2" \
    && tar xvf eigen-3.3.2.tar.bz2 \
    && cd eigen-3.3.2 && mkdir build && cd build && cmake .. \
    && make -j$(nproc) && make install \
    && cd / && rm -rf /tmp/eigen-3.3.2 && rm /tmp/eigen-3.3.2.tar.bz2   

# Pangolin
RUN cd /tmp && git clone https://github.com/stevenlovegrove/Pangolin.git \
    && cd Pangolin && git checkout v0.6 && mkdir build && cd build && cmake .. \
    && make -j$(nproc) && make install \
    && cd / && rm -rf /tmp/Pangolin

# VTune Profiling
# RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
#     | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
# RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
#     | tee /etc/apt/sources.list.d/oneAPI.list
# RUN apt update &&\
#     apt-get install -y \
#         libnss3 \
#         libatk-bridge2.0-0 \
#         libgtk-3-0 \
#         libasound2 \
#         intel-oneapi-vtune \
#     && rm -rf /var/lib/apt/lists/*

#RUN source /opt/intel/oneapi/vtune/latest/env/vars.sh