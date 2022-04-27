from argparse import ArgumentParser
import os
import pathlib
import uuid

from flask import Flask, flash, redirect, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from painter import Painter


UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

APP = Flask(__name__)
APP.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

PAINTER = Painter()


def make_html_styles():
    html = ''
    for i_path, path_style in enumerate(pathlib.Path('styleimages').iterdir()):
        id_style = path_style.stem
        html_style = f'''
            <div>
                <input type="radio" name="style" 
                       id="{id_style}" value="{id_style}"
                       {"checked" if i_path == 0 else ""}>
                <label for="{id_style}">
                    <img src="styleimages/{path_style.name}" width="100">
                </label>
            </div>
            '''
        html += html_style
    return html


@APP.route('/styleimages/<name>')
def styleimage(name):
    return send_from_directory('styleimages', name)


def allowed_file(filename):
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def download_file(style, name):
    way_style = f"styleimages/{style}.jpg"
    way_content = APP.config["UPLOAD_FOLDER"] + '/' + name
    alpha = 1
    basename_result = name + '.result.jpg'
    way_result = APP.config["UPLOAD_FOLDER"] + '/' + basename_result
    PAINTER.paint(way_style, way_content, alpha, way_result)
    return send_from_directory(APP.config["UPLOAD_FOLDER"], basename_result)


@APP.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']

        if file:
            filename = str(uuid.uuid4()) + '.jpg'
            file.save(os.path.join(APP.config['UPLOAD_FOLDER'], filename))

            style = request.headers.get('X-Style')
            if style is None:
                style = request.form['style']

            return download_file(
                style=style,
                name=filename,
            )

    return f'''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
            {make_html_styles()}
            <input type=file name=image>
            <input type=submit value=Upload>
        </form>
    '''
