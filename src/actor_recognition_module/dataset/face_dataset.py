import abc
import os
import random
from abc import abstractmethod
from mtcnn import mtcnn

import numpy
from keras_preprocessing.image import load_img, img_to_array
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.utils import np_utils
from collections import defaultdict
from util import constant
from util.common import Common


class FaceDataSet(metaclass=abc.ABCMeta):

    def __init__(self, path,  extension_list):
        self.path = path
        self.ext_list = extension_list
        self.objects = defaultdict(list)
        self.labels = []
        self.label_names = []
        self.obj_validation = defaultdict(list)
        self.labels_validation = []
        self.number_labels = 0

    @abstractmethod
    def get_data(self, training_split=0.8):
        """
        Obtiene los objetos (imagenes) y etiquetas del dataset y los divide en subsets de entrenamiento y validacion.\n
        ---
        Parametros:\n
        training_split: float
            Proporcion del dataset que se utilizará para entrenar. La proporción de validación será (1 - training_split)
        """
        pass

    @abstractmethod
    def split_training_set(self, training_split):
        """
        Divide el dataset en subsets de entrenamiento y validacion.\n
        ---
        Parametros:\n
        training_split: float
            Proporcion del dataset que se utilizará para entrenar. La proporción de validación será (1 - training_split)
        """
        pass

    @abstractmethod
    def print_dataSet(self):
        """
        Imprime un resumen del dataset.\n
        """
        print(self.objects.values)
        print(self.objects.keys)

    @abstractmethod
    def process_label(self, obj):
        """
        Obtiene la etiqueta para un objeto del dataset.\n
        ---
        Parametros:\n
        obj: Any
            Objeto utilizado para extraer la etiqueta.
        """
        pass

    def check_ext(self, file_path):
        """
        Chequea si la extensión del archivo está entre las permitidas.\n
        ---
        Parametros:\n
        file_path: str
            Path del archivo.
        """
        for ext in self.ext_list:
            if file_path.endswith(ext):
                return True
        return False