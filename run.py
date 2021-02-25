#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2021, Linagora, Ilyes Rebai
# Email: irebai@linagora.com


from flask import Flask, request, abort, Response, json
from gevent.pywsgi import WSGIServer
from logging.config import fileConfig
from SpeakerDiarization import *


# Begin Service configuration
app = Flask("__speaker-diarization-worker__")
app.debug = True
fileConfig('logging.cfg')
worker = SpeakerDiarization()

SERVICE_PORT=80
SWAGGER_URL = '/api-doc'
if 'SWAGGER_PATH' in os.environ:
    SWAGGER_PATH = os.environ['SWAGGER_PATH']
# End Service configuration


# API
@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return "1", 200

@app.route('/', methods=['POST'])
def run():
    try:
        app.logger.info('[POST] New user entry - Speaker diarization')
        if 'file' in request.files.keys():
            file = request.files['file']
            result = worker.run(file)
            response = worker.format_response(result)
        else:
            return 'No audio file was uploaded', 400
        return response, 200
    except ValueError as error:
        app.logger.error(error)
        return str(error), 400
    except Exception as e:
        app.logger.error(e)
        return 'Server Error', 500

# Rejected request handlers
@app.errorhandler(405)
def method_not_allowed(error):
    return 'The method is not allowed for the requested URL', 405

@app.errorhandler(404)
def page_not_found(error):
    return 'The requested URL was not found', 404

@app.errorhandler(500)
def server_error(error):
    app.logger.error(error)
    return 'Server Error', 500



if __name__ == '__main__':
    try:
        # Run SwaggerUI
        if worker.swaggerUI(app):
            app.logger.info("The SwaggerUI is started successfully on port 80")
        else:
            app.logger.warning("The SwaggerUI is not started! Please check the SWAGGER_PATH volume.")

        # Run server
        app.logger.info("Starting the speaker diarization service on port 80")
        http_server = WSGIServer(('', SERVICE_PORT), app)
        http_server.serve_forever()

    except Exception as e:
        app.logger.error(e)
        exit(e)
