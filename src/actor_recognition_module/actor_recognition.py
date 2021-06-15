import pickle
import os
import cv2
import imutils

from util import constant as app_constant
from actor_recognition_module.util import constant as facerec_constant

################################################################################################

backend_folder = os.getcwd()
path_encodings = os.path.join(os.path.dirname(__file__), 'encodings', 'encodings.pickle')

################################################################################################

def load_model():
    return pickle.loads(open(path_encodings, "rb").read())

def get_n_frames(path_video, interval = 1):
    frames = []
    
    video = cv2.VideoCapture(path_video)
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    success, image = video.read()
    
    count = 0
    while success:
        frames.append(image)
        count += interval
        video.set(1, int(count * frame_rate))
        success, image = video.read()
    
    return frames

def get_actors_in_video(path_video, detection_method='hog', verbose=False, threshold=0.6):
    if verbose:
        print("[REC-ACT] Iniciando reconocimiento de actores en video...")

    names = set()

    # load the known faces and embeddings
    if verbose:
        print("[REC-ACT] Cargando encodings...")

    data = pickle.loads(open(path_encodings, "rb").read())

    # initialize the video stream and pointer to output video file, then
    # allow the camera sensor to warm up
    if verbose:
        print("[REC-ACT] Procesando video...")
    vs = cv2.VideoCapture(path_video)

    faceRec = FaceRec()

    # loop over frames from the video file stream
    while True:

        # grab the frame from the threaded video stream
        (grabbed, frame) = vs.read()

        # If frame wasn't grabbed, we've reached the end
        if not grabbed:
            break

        # convert the input frame from BGR to RGB then resize it to have
        # a width of 750px (to speedup processing)
        image_rgb_resized = imutils.resize(frame, width=750)

        # based on user args select fast kdtree based nn or linear search
        names.update(faceRec.getAllFacesInImage(image_rgb_resized, detection_method, False,
                                                data[constants.KNOWN_ENCODINGS], data[constants.ENCODING_STRUCTURE],
                                                data[constants.KNOWN_NAMES], threshold))

    if verbose:
        print("[REC-ACT] Caras identificadas.")

    return list(names)

################################################################################################