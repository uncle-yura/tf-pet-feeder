# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile
ARG BUILD_FROM
FROM $BUILD_FROM

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN \
    set -x \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        nginx \
        python3 \
        python3-pip \
        build-essential \
    && apt-get clean \
    && pip3 install --upgrade pip setuptools wheel \
    && rm -f -r /var/lib/apt/lists/* \
    && find /usr/local \
        \( -type d -a -name test -o -name tests -o -name '__pycache__' \) \
        -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) \
        -exec rm -rf '{}' +

COPY requirements.txt .

RUN \
    pip3 install \
        --no-cache-dir \
        -r requirements.txt

ARG BUILD_ARCH
RUN if [ "$BUILD_ARCH" = "amd64" ]; then \
        pip3 install https://tf.novaal.de/core2/tensorflow-2.8.0-cp39-cp39-linux_x86_64.whl ; \
    elif [ "$BUILD_ARCH" = "aarch64" ]; then  \
        pip3 install tensorflow ;  \
    fi

# Copy root filesystem
COPY rootfs /
