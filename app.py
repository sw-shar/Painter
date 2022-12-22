import base64
import binascii
import flask
import jinja2
import json
import os
import pathlib
import uuid
import zipfile

import microbook

JINJA_ENVIRONMENT = jinja2.Environment(loader=jinja2.FileSystemLoader("templates/"))
INDEX_TEMPLATE = JINJA_ENVIRONMENT.get_template("index.html")

APP = flask.Flask(__name__)


def make_answer(query):
    try:
        rows = microbook.exit_sql(query)
    except Exception as exc:
        return {'error': str(exc)}

    name, price, image_url = rows[0]
    return {'name': name, 'price': price, 'image_url': image_url}


@APP.route('/', methods=['GET', 'POST'])
def upload_file():
    query = None
    answer = {}
    if flask.request.method == 'POST':
        query = flask.request.form['query']
        answer = make_answer(query)

    #return str(answer)
    return INDEX_TEMPLATE.render(query=query, **answer)
