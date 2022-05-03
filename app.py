import base64
import binascii
import json
import os
import pathlib
import uuid
import zipfile

from flask import Flask, request, send_from_directory, send_file

from painter import Painter


UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

ERROR_400 = ("bad request", 400)
ERROR_503 = ("модель не смогла обработать данные", 503)

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


def download_file(style, path_content, *, is_json=True, are_metrics=False,
                  jobid):
    path_style = f'styleimages/{style}.jpg'
    alpha = 1

    path_result = f'{path_content}.result-{jobid}.jpg'
    # Note: `jobid` is to avoid name clashes.

    try:
        metrics = PAINTER.paint(path_style, path_content, alpha, path_result)
    except Exception:
        return ERROR_503

    if not is_json:
        return send_file(path_result)

    with open(path_result, 'rb') as fileobj:
        image_bytes = fileobj.read()

    image_base64_bytes = base64.b64encode(image_bytes)
    image_base64_str = image_base64_bytes.decode()
    dic = {'image': image_base64_str}
    if are_metrics:
        dic['metrics'] = metrics
    return dic


@APP.route('/forward', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        jobid = str(uuid.uuid4())
        basename = jobid + '.jpg'
        filename = os.path.join(APP.config['UPLOAD_FOLDER'], basename)
        is_json = request.content_type.startswith('application/json')

        if is_json:
            data = request.get_data()
            try:
                dic = json.loads(data.decode('utf-8'))
            except (UnicodeDecodeError, json.decoder.JSONDecodeError):
                return ERROR_400

            try:
                style = dic['style']
                image_base64 = dic['image']
            except KeyError:
                return ERROR_400

            try:
                image_bytes = base64.b64decode(image_base64)
            except binascii.Error:
                return ERROR_400

            with open(filename, 'wb') as fileobj:  # write, binary
                fileobj.write(image_bytes)
        else:
            style = request.headers.get('X-Style')
            if style is None:
                try:
                    style = request.form['style']
                except KeyError:
                    return ERROR_400

            try:
                image_file = request.files['image']
            except KeyError:
                return ERROR_400

            image_file.save(filename)

        return download_file(
            style=style,
            path_content=APP.config["UPLOAD_FOLDER"] + '/' + basename,
            is_json=is_json,
            jobid=jobid,
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


@APP.route('/metadata')
def get_metadata():
    return PAINTER.get_metadata()


def evaluate_files(*, style, fileid2filename, are_metrics, jobid):
    return {fileid: download_file(style, filename, are_metrics=are_metrics,
                                  jobid=jobid)
            for fileid, filename in fileid2filename.items()}


@APP.route('/forward_batch', methods=['POST'])
def forward_batch():
    return evaluate_core(are_metrics=False)


@APP.route('/evaluate', methods=['POST'])
def evaluate():
    return evaluate_core()


def evaluate_core(are_metrics=True):
    jobid = str(uuid.uuid4())
    basename = jobid + '.zip'
    filename = os.path.join(APP.config['UPLOAD_FOLDER'], basename)
    path_dir = os.path.join(APP.config['UPLOAD_FOLDER'], jobid)
    os.mkdir(path_dir)

    try:
        style = request.headers['X-Style']
    except KeyError:
        return ERROR_400

    try:
        zip_file = request.files['zip_in']
    except KeyError:
        return ERROR_400

    zip_file.save(filename)

    with zipfile.ZipFile(filename, 'r') as file_zip:
        file_zip.extractall(path_dir)

    len_prefix = len(path_dir) + 1
    fileid2filename = {str(path)[len_prefix:]: str(path)
                       for path in pathlib.Path(path_dir).glob('**/*')
                       if not path.is_dir()}

    return evaluate_files(
        style=style,
        fileid2filename=fileid2filename,
        are_metrics=are_metrics,
        jobid=jobid,
    )
