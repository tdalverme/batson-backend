from PIL import Image
from mtcnn import MTCNN
from numpy import asarray
from face_detection.detector import Detector
from util import constant

class MTCnnDetector(Detector):

    def __init__(self):
        self.detector = MTCNN()

    def load_img(self, image_path):
        return super().load_img(image_path)

    def process_image(self):
        faces = self.__detect_face()
        resized_face_list = []
        for f in faces:
            extracted_face = self.__extract_face(f)
            resized_face = self.__resize_img_to_face(extracted_face)
            resized_face_list.append(resized_face)
        return resized_face_list

    def __detect_face(self):
        return self.detector.detect_faces(self.image)

    def __extract_face(self, face):
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height

        x1, x2, y1, y2 = max([0, x1]), max([0, x2]), max([0, y1]), max([0, y2])

        return self.image[y1:y2, x1:x2]

    def __resize_img_to_face(self, face):
        image = Image.fromarray(face)
        image = image.resize((constant.IMG_WIDTH, constant.IMG_HEIGHT))
        return asarray(image)