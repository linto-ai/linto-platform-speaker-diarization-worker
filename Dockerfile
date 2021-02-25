FROM ubuntu:18.04
LABEL maintainer="irebai@linagora.com"

RUN apt-get update &&\
    apt-get install -y \
    python3     \
    python3-pip \
    nano \
    sox  \
    ffmpeg \
    software-properties-common \
    wget \
    curl && \
    apt-get clean

# Install main service packages
RUN pip3 install flask flask-swagger-ui gevent

# Install pyBK dependencies
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 10 && \
    export LLVM_CONFIG=/usr/bin/llvm-config-10 && \
    pip3 install numpy && \
    pip3 install librosa webrtcvad scipy sklearn


# Define the main folder
WORKDIR /usr/src/speaker-diarization

# Copy main functions
COPY docker-entrypoint.sh logging.cfg SpeakerDiarization.py run.py ./
COPY pyBK/diarizationFunctions.py pyBK/diarizationFunctions.py

# logs output directory
RUN mkdir -p /opt/logs

EXPOSE 80

HEALTHCHECK CMD curl http://localhost/healthcheck || exit 1

# Entrypoint handles the passed arguments
ENTRYPOINT ["./docker-entrypoint.sh"]