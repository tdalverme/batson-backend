from os.path import join, dirname

# PATHS
BASE_PATH = 'D:\\Facultad\\Proyecto Final de Carrera\\batson-pruebas'
FACEREC_MODEL_PATH = join(BASE_PATH, 'src\\actor_recognition_module\\models\\encodings_506.pickle')
DATASET_PATH = join(BASE_PATH, 'resources\\dataset')
TEST_FOLDER_PATH = join(BASE_PATH, 'resources\\test')

# PARAMETROS DEL MODELO
ENC_LIST = 'list'
KNOWN_ENCODINGS = 'known_encodings'
KNOWN_NAMES = 'known_names'
ID_UNKNOWN = 'Unknown'
ENCODING_STRUCTURE = 'encoding_structure'
TOLERANCE = 0.5

# PARAMETROS DE IMAGENES/VIDEO
IMG_WIDTH = 224
IMG_HEIGHT = 224