from PIL import Image
import face_recognition
from face_detection.detector import Detector
from util import constant

class FaceRecDetector(Detector):

    def __init__(self):
        super().__init__()

    def load_img(self, image_path):
        """
        Devuelve una imagen.\n
        ---
        Parametros:\n
        image_path: str
            Path del archivo imagen.
        ---
        Return: ndarray
            Devuelve la imagen en forma de array.
        """
        return super().load_img(image_path)

    def process_image(self):
        """
        Devuelve una lista de las caras encontradas en la imagen.\n
        ---
        Return: list(Image)
            Devuelve una lista de imágenes con las caras en la imagen.
        """
        boxes = face_recognition.face_locations(self.image, model='hog')

        images = list()
        for box in boxes:
            img = self.image[box[0]:box[2], box[3]:box[1]] 
            images.append(img)

        res = list()
        for image in images:
            image = Image.fromarray(image)
            image = image.resize((constant.IMG_WIDTH, constant.IMG_HEIGHT))
            res.append(image)

        return res