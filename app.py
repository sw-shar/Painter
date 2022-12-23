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


def make_method_rows(query):
    """
    >>> make_method_rows("400914-00212")  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    {'method': 'sql', 'rows': [('Насос основной', 'Doosan', 'dx225', 220000, ...)]}

    >>> make_method_rows("r160lc-7")  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    {'method': 'predict', 
     'rows': [('Насос основной', 'Hyundai', 'r160lc-9', 150000, ...), 
              ('Редуктор хода', 'Hyundai', 'r160lc-9', 180000, ...)]}
    """
    method = "sql"
    rows = microbook.exit_sql(query)
    if not rows:
        method = 'predict'
        rows = microbook.predict_marka_model(query)
        if not rows:
            return {"error": 'not found'}
    return {'method': method, 'rows': rows}


def make_answer(query):
    try:
        method_rows = make_method_rows(query)
    except Exception as exc:
        return {"error": str(exc)}

    method = method_rows['method']
    rows = method_rows['rows']

    name, marka, model, price, image_url = rows[0]
    print(rows)
    return {
        "name": name,
        "marka": marka,
        "model": model,
        "price": price,
        "image_url": image_url,
    }


@APP.route("/", methods=["GET", "POST"])
def upload_file():
    query = None
    answer = {}
    if flask.request.method == "POST":
        query = flask.request.form["query"]
        answer = make_answer(query)
        # TODO: logging

    # return str(answer)
    return INDEX_TEMPLATE.render(query=query, **answer)
