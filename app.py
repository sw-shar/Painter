from argparse import ArgumentParser
import os

from flask import Flask, flash, redirect, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from painter import Painter


UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

APP = Flask(__name__)
APP.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

PAINTER = Painter()


def allowed_file(filename):
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@APP.route('/uploads/<name>')
def download_file(name):
    # PAINTER.paint(name, name + '.result.jpg')
    return send_from_directory(APP.config["UPLOAD_FOLDER"], name)


@APP.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(APP.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('download_file', name=filename))
    return f'''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <p>PAINTER = {PAINTER}</p>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
