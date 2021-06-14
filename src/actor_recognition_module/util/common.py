import numpy
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

from util import constant


class Common:

    @staticmethod
    def reshape_transform_data(data):
        data = numpy.array(data, dtype=object)
        result = Common.reshape_data(data)
        return Common.to_float(result)

    @staticmethod
    def reshape_data(data):
        return data.reshape(data.shape[0], constant.IMG_WIDTH, constant.IMG_HEIGHT, 3)

    @staticmethod
    def reshape_from_img(image):
        return image.reshape((constant.IMG_WIDTH, constant.IMG_HEIGHT, 3))

    @staticmethod
    def to_float(value):
        return value.astype('float32')/255

    @staticmethod
    def get_sgd_optimizer():
        return SGD(lr=0.01, decay=1e-65, momentum=0.92, nesterov=True)