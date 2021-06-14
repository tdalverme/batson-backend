import os
import random
import numpy as np
import cv2
from dataset.face_dataset import FaceDataSet
from util import constant

class ActorsDataSet(FaceDataSet):

    def __init__(self, path, ext_list):
        super().__init__(path, ext_list)

    def get_data(self, training_split=0.8):
        for r, d, f in os.walk(self.path):
            for file in f:
                actor_name = self.process_label(file)
                if super().check_ext(file):
                    img_abs_path = os.path.abspath(os.path.join(self.path, actor_name, file))

                    try:
                        img = cv2.imread(img_abs_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        self.objects[actor_name].append(img)
                    except Exception as ex:
                        print("Error %s" % ex)
                        pass

        self.split_training_set(training_split)

    def split_training_set(self, training_split=0.8):
        for key in self.objects.keys():
            self.obj_validation[key] = random.sample(self.objects[key], int(len(self.objects[key]) * (1 - training_split)))
            self.objects[key] = [item for item in self.objects[key] if not any(np.array_equal(item, x) for x in self.obj_validation[key])]
    
    def print_dataSet(self):
        training_obj = 0
        validation_obj = 0

        for k, v in self.objects.items():
            training_obj += v.__len__()
        
        for k, v in self.obj_validation.items():
            validation_obj += v.__len__()
        
        print("Training objects: %d" % training_obj)
        print("Validation objects: %d" % validation_obj)
        print("Labels: ", end='')
        print(list(self.obj_validation.keys()))

    def process_label(self, file_path):
        """
        Obtiene la etiqueta para un objeto del dataset.\n
        ---
        Parametros:\n
        file_path: str
            Path del archivo de donde se extraerá la etiqueta.
        ---
        Return: str
            Etiqueta del objeto.
        """
        return file_path.split('-')[0]