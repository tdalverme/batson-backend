import functools
import flask
from flask import request
import tempfile
import os
import json
import multiprocessing as mp

from werkzeug.utils import secure_filename
from actor_recognition_module.actor_recognition import get_n_frames
from actor_recognition_module.util import constant as facerec_constant
from actor_recognition_module.face_rec.model.facerec_model import FaceRecModel
from audio_module.audio_utils import *
from audio_module.ponderation_by_audio import *
from actor_recognition_module.imdb_utils import *
from util import constant as app_constant
from util import utils

###############################################################################################
    
app = flask.Flask(__name__)
model = FaceRecModel('cnn')

###############################################################################################

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in [*app_constant.ALLOWED_EXTENSIONS_IMAGE, *app_constant.ALLOWED_EXTENSIONS_VIDEO]

def process_video(path_video):
    # obtener una lista de frames del video
    frames = get_n_frames(path_video)

    if app_constant.VERBOSE:
        print("[INFO] Cantidad de frames a analizar: %d" % len(frames))

    if app_constant.VERBOSE:
        print("[INFO] Reconociendo actores en frames...")

    # se realiza una prediccion para cada imagen usando multiprocesamiento
    predict_function = functools.partial(model.predict_from_img)
    with mp.Pool(2) as pool:
        names_list = pool.map(predict_function, frames)

    names_list = utils.flatten_list(names_list)

    # se crea un diccionario (actor, nro_ocurrencias)
    # si se predice un actor en más de un frame es más probable que la prediccion sea correcta
    d = dict()
    d = d.fromkeys(names_list, 0)

    total = 0
    for name in names_list:
        if name != facerec_constant.ID_UNKNOWN:
            d[name] += 1
            total += 1
    
    for k, v in d.items():
        d[k] = "%.2f%%" % (v / total * 100)

    return d
    

def process_image(path_image, verbose=False):
    if app_constant.VERBOSE:
        print("[INFO] Reconociendo actores...")
    actors = model.predict(path_image)
    return actors

@app.route('/', methods=['GET'])
def home():
    return "<h1>TEST API</p>"

@app.route('/upload-file/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Error {}: No file part".format(app_constant.NO_FILE_PART), app_constant.NO_FILE_PART

        uploaded_file = request.files['file']
        
        if uploaded_file.filename == '':
            return "Error {}: File not selected".format(app_constant.FILE_NOT_SELECTED), app_constant.FILE_NOT_SELECTED

        if uploaded_file and allowed_file(uploaded_file.filename):
            filename = secure_filename(uploaded_file.filename)
            uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        if filename.rsplit('.')[1].lower() in app_constant.ALLOWED_EXTENSIONS_IMAGE:
            if app_constant.VERBOSE:
                print("[INFO] Procesando imagen...")
            res = process_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            if app_constant.VERBOSE:
                print("[INFO] Se termino de procesar la imagen")
            return json.dumps(res)

        if filename.rsplit('.')[1].lower() in app_constant.ALLOWED_EXTENSIONS_VIDEO:
            if app_constant.VERBOSE:
                print("[INFO] Procesando video...")
            res = process_video(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            if app_constant.VERBOSE:
                print("[INFO] Se termino de procesar el video")
            return json.dumps(res)

    else:
        return "Error %d Method Not Allowed" % app_constant.METHOD_NOT_ALLOWED, app_constant.METHOD_NOT_ALLOWED

###############################################################################################

if __name__ == '__main__':
    app.config["DEBUG"] = True
    app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

    model.load_from_file(facerec_constant.FACEREC_MODEL_PATH)

    app.run(debug=True, use_reloader=False)