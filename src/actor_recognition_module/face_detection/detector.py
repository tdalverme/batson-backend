from abc import abstractmethod
from matplotlib import pyplot
from skimage import io

class Detector:

    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def process_image(self):
        pass

    def load_img(self, image_path):
        self.image = io.imread(image_path, as_gray=False)

    def plot_face(self, face):
        """
        Muestra la cara en la imagen.\n
        ---
        Parametros:\n
        face: str
            Path del archivo imagen.
        ---
        Return: ndarray
            Devuelve la imagen en forma de array.
        """
        pyplot.imshow(face)
        pyplot.show()