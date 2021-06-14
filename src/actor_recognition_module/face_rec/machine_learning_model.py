import abc
from abc import abstractmethod

class MLModel(metaclass=abc.ABCMeta):

    def __init__(self, dataSet=None):
        if dataSet is not None:
            self.objects = dataSet.objects
            self.labels = dataSet.labels
            self.label_names = dataSet.label_names
            self.obj_validation = dataSet.obj_validation
            self.labels_validation = dataSet.labels_validation
            self.number_labels = dataSet.number_labels
            self.dataset_path = dataSet.path
        self.init_model()

    @abstractmethod
    def init_model(self):
        """
        Inicializa el modelo de reconocimiento de caras.\n
        """
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, object):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def load_from_file(self, path):
        pass

    @abstractmethod
    def save(self, path):
        pass

    def evaluate(self):
        score = self.get_model().evaluate(self.obj_validation, self.labels_validation, verbose=0)
        print("%s: %.2f%%" % (self.get_model().metrics_names[1], score[1] * 100))