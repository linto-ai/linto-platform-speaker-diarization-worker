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


# Define the main folder
WORKDIR /usr/src/speaker-diarization

# Copy main functions
COPY docker-entrypoint.sh .
COPY logging.cfg .
COPY run.py .

# logs output directory
RUN mkdir -p /opt/logs

EXPOSE 80

HEALTHCHECK CMD curl http://localhost/healthcheck || exit 1

# Entrypoint handles the passed arguments
ENTRYPOINT ["./docker-entrypoint.sh"]