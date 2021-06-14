import time
import os
from dataset.datasets.actors_dataset import ActorsDataSet
from face_rec.model.facerec_model import FaceRecModel
from util import constant

if __name__ == "__main__":
    ext_list = ['jpg', 'jpeg', 'png']
    # Set up dataSet
    dataSet = ActorsDataSet(constant.DATASET_PATH, ext_list)
    dataSet.get_data()
    dataSet.print_dataSet()

    model = FaceRecModel('cnn', dataSet)

    #model.train(multiprocessing=True)
    #model.save(constant.FACEREC_MODEL_PATH)
    model.load_from_file(constant.FACEREC_MODEL_PATH)

    for r, d, f in os.walk(constant.TEST_FOLDER_PATH):
        for file in f:
            path = os.path.join(r, file)
            start = time.time()
            print(model.predict(path))
            end = time.time()
            print("================ {} seconds ================".format(end - start))

    model.evaluate()