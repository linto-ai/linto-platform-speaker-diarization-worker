# linto-platform-speaker-diarization-worker
Speaker diarization worker. The process of segmenting and co-indexing speech signals by speaker.


# Develop

## Installation

### Packaged in Docker
First of all, you need to install docker and docker-compose on your machine in order to build and run this service.

To start the LinSD service on your local machine or your cloud, you need first to download the source code and set the environment file, as follows:

```bash
git clone https://github.com/linto-ai/linto-platform-speaker-diarization-worker
cd linto-platform-speaker-diarization-worker
git submodule update --init
mv .envdefault .env
```

Then, you can simply downalod the pre-built image from docker-hub

```bash
docker pull lintoai/linto-platform-speaker-diarization-worker:latest
```


Or, to build the docker image, by executing:

```bash
docker build -t lintoai/linto-platform-speaker-diarization-worker:latest .
```

## Execute
In order to run the service, you have to configure the environment file `.env`:

    LOGS_PATH=/path/to/save/the/log/file

Thereafter, you have only to execute `docker-compose up`

of using the docker run command:

```bash
cd linto-platform-speaker-diarization-worker
docker run -p 8888:80 -v /full/path/to/save/the/log/file:/opt/logs -v /full/path/to/linto-platform-speaker-diarization-worker/document/swagger.yml:/opt/swagger/swagger.yml lintoai/linto-platform-speaker-diarization-worker:latest
```

### Run Example Applications
To run a test, you can use swagger interface: `localhost:8888/api-doc/`