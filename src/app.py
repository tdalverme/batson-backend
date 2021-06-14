import flask
from flask import request
import tempfile
import sys
import os
import json
import multiprocessing as mp

from werkzeug.utils import secure_filename
from actor_recognition_module.util import constant as facerec_constant
from actor_recognition_module.face_rec.model.facerec_model import FaceRecModel
from audio_module.audio_utils import *
from audio_module.ponderation_by_audio import *
from actor_recognition_module.imdb_utils import *
from util import constant as app_constant

###############################################################################################
    
app = flask.Flask(__name__)
model = FaceRecModel('cnn')

###############################################################################################

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in [*app_constant.ALLOWED_EXTENSIONS_IMAGE, *app_constant.ALLOWED_EXTENSIONS_VIDEO]

def process_video(path_video, verbose=False):
    # separar audio de videos
    path_audio = separate_audio(path_video)

    # ejecutar paralelamente reconocimiento de audio y reconocimiento de actores
    pool = mp.Pool(1)
    pool2 = mp.Pool(1)

    pool.apply_async(audio_to_text, [path_audio, True])
    actors = pool2.apply_async(get_actors_in_video, [path_video, 'cnn', True])

    # hay que esperar a que se haya terminado de obtener los actores
    pool2.close()
    pool2.join()

    # iniciar busqueda de peliculas con los actores encontrados
    # si el reconocimiento de audio no finalizo se ejecuta en paralelo
    pool2 = mp.Pool(1)

    movies_ids = pool2.apply_async(get_movies_with_actors, [actors, True])

    # hay que esperar a que se haya terminado de obtener las peliculas
    pool2.close()
    pool2.join()

    # obtener los plots de cada pelicula
    plots = get_plots(movies_ids.get(), verbose = verbose)
    
    # hay que esperar a que se haya terminado de reconocer el audio
    pool.close()
    pool.join()
    
    os.remove(path_audio)

    # obtener similitudes de las peliculas con el audio reconocido
    sims = movie_similarity_by_audio(movies_ids.get(), plots, verbose = verbose)
    return sims

def process_image(path_image, verbose=False):
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
            res = process_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return json.dumps(res)

        if filename.rsplit('.')[1].lower() in app_constant.ALLOWED_EXTENSIONS_VIDEO:
            res = process_video(os.path.join(app.config['UPLOAD_FOLDER'], filename), True)
            return json.dumps(res)

    else:
        return "Error %d Method Not Allowed" % app_constant.METHOD_NOT_ALLOWED, app_constant.METHOD_NOT_ALLOWED

###############################################################################################

if __name__ == '__main__':
    app.config["DEBUG"] = True
    app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

    model.load_from_file(facerec_constant.FACEREC_MODEL_PATH)

    app.run(debug=True, use_reloader=False)