#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2021, Linagora, Ilyes Rebai
# Email: irebai@linagora.com


from flask import Flask, request, abort, Response, json
from gevent.pywsgi import WSGIServer
from logging.config import fileConfig



# Service configuration start
app = Flask("__speaker-diarization-worker__")

fileConfig('logging.cfg')

SERVICE_PORT=80
# Service configuration end


# API
@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return "1", 200

@app.route('/', methods=['GET'])
def speakerDiarization():
    app.logger.info("start Speaker diarization")
    return "1", 200


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
        app.debug = True
        app.logger.info("Starting the speaker diarization service on port 80")
        http_server = WSGIServer(('', SERVICE_PORT), app)
        http_server.serve_forever()

    except Exception as e:
        app.logger.error(e)
        exit(e)
