import functools
import pickle
import cv2
import face_recognition
import multiprocessing as mp
from actor_recognition_module.face_rec.machine_learning_model import MLModel
from actor_recognition_module.util import constant

class FaceRecModel(MLModel):

    def __init__(self, detection_method, dataSet=None):
        self.known_encodings = list()
        self.known_names = list()
        self.detection_method = detection_method
        super().__init__(dataSet)

    def init_model(self):
        pass

    def get_model(self):
        """
        Devuelve el modelo de reconocimiento de caras.\n
        ---
        Return: FaceRecModel
        """
        return self.model

    @staticmethod
    def encode_faces(object, label, detection_method):
        """
        Codifica las caras en el objeto de acuerdo al método de detección especificado.\n
        ---
        Parametros:
            object: ndarray
                Imagen en forma de array.\n
            label: str
                Etiqueta del objeto.\n
            detection_metod: 'hog' | 'cnn'
                Método de detección de caras.\n
                hog: Histogram of Oriented Gradient
                cnn: Convolutional Neural Network
        ---
        Return: list(ndarray), list(str)
            Devuelve una lista de las codificaciones de las caras con las etiquetas respectivas.
        """
        known_encodings = list()
        known_names = list()
        try:
            boxes = face_recognition.face_locations(object, model=detection_method)
            encodings = face_recognition.face_encodings(object, boxes)

            for encoding in encodings:
                known_encodings.append(encoding)
                known_names.append(label)
                
            return known_encodings, known_names
        except Exception as ex:
            raise Exception("Exception while encoding image of %s : %s" % (label, ex))
    
    def train(self, cores = 2, multiprocessing=False):
        """
        Entrena el modelo.\n
        ---
        Parametros:
            cores: int
                Cantidad de núcleos del CPU utilizados para el multiprocesamiento.\n
            multiprocessing: bool
                Indica si se utiliza multiprocesamiento o no.\n
        """
        for key, val in self.objects.items():
            if multiprocessing:
                encode_images_with_detection_method = functools.partial(FaceRecModel.encode_faces, label=key, detection_method=self.detection_method)
            
                # loop over the images using batching for multiprocessing
                for i in range(0, len(val), batch):
                    
                    # using pool for parallelism
                    obj_batch = val[i:i + batch]
                    with mp.Pool(cores) as pool:
                        encodings_names_list = pool.map(encode_images_with_detection_method, obj_batch)

                    encodings_names_list = filter(lambda t: len(t[0]) > 0 and len(t[1]) > 0, encodings_names_list)
                    
                    for (encodings, names) in encodings_names_list:
                        self.known_encodings.append(encodings)
                        self.known_names.append(names)
            else:
                for object in val:
                    encodings, names = FaceRecModel.encode_faces(object, key, self.detection_method)

                    for encodings, names in zip(encodings, names):
                        self.known_encodings.append(encodings)
                        self.known_names.append(names)

    def predict(self, img_path):
        """
        Predice la etiqueta o clase de una imagen.\n
        ---
        Parametros:
            img_path: str
                Path de la imagen.\n
        ---
        Return: list(str)
            Devuelve una lista de las etiquetas o clases de las caras en la imagen.
        """
        boxes = list()
        encodings = list()

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(img, model='hog')
        encodings = face_recognition.face_encodings(img, boxes)

        names = list()
        if encodings:
            names = self.__linear_search(encodings)

        return names

    def __linear_search(self, query_encodings):
        """
        Busca la etiqueta o clase más similar a query_encodings.\n
        ---
        Parametros:
            query_encodings: list(ndarray)
                Lista de los encodings a consultar.\n
        ---
        Return: list(str)
            Devuelve una lista de las etiquetas o clases más similares.
        """
        names = list()
        
        for encoding in query_encodings:
            matches = face_recognition.compare_faces(self.known_encodings, encoding, tolerance=constant.TOLERANCE)
            name = constant.ID_UNKNOWN

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                for i in matchedIdxs:
                    name = self.known_names[i]
                    counts[name] = counts.get(name, 0) + 1

                name = max(counts, key=counts.get)

            names.append(name)
            
        return names

    def evaluate(self):
        """
        Evalúa el modelo e imprime la precisión del mismo.
        """
        corrects = 0
        total = 0

        for key, objects in self.obj_validation.items():
            for val in objects:
                boxes = face_recognition.face_locations(val, model='hog')
                encodings = face_recognition.face_encodings(val, boxes)

                names = list()
                if encodings:
                    names = self.__linear_search(encodings)
                
                if names and names[0] == key:
                    corrects += 1

                total += 1

        accuracy = (corrects / float(total)) * 100
        print("Accuracy: %.2f%%" % accuracy)

    def save(self, path):
        """
        Guarda el modelo a un archivo.\n
        ---
        Parametros:
            path: str
                Path del archivo donde se guardará el modelo.\n
        """
        encoding_structure = constant.ENC_LIST

        data =  {   constant.KNOWN_ENCODINGS: self.known_encodings,
                    constant.KNOWN_NAMES: self.known_names,
                    constant.ENCODING_STRUCTURE: encoding_structure }

        f = open(path, "wb")
        f.write(pickle.dumps(data))
        f.close()

    def load_from_file(self, path):
        """
        Cargar el modelo desde un archivo.\n
        ---
        Parametros:
            path: str
                Path del archivo de donde se cargará el modelo.\n
        """
        data = pickle.loads(open(path, "rb").read())
        self.known_encodings, self.known_names = data[constant.KNOWN_ENCODINGS], data[constant.KNOWN_NAMES]