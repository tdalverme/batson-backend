# run python test_server.py

import base64
import time
import os
import requests
import json
from actor_recognition_module.util import constant

#############################################################################

file_path = os.path.join(constant.TEST_FOLDER_PATH, 'avengers_vision.mp4')

#############################################################################

#with open(os.path.join(path_base, 'resources', 'ultron.mp4'), "rb") as videoFile:
#    video_to_base64 = base64.b64encode(videoFile.read())

url = "http://127.0.0.1:5000/upload-file/"

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36',
    'Content-type': 'application/json'
}

files = {
    'file': open(file_path, 'rb')
}

start = time.time()
res = requests.post(url, files=files)
print("%.2f" % (time.time() - start), " seconds")

if file_path.rsplit('.')[1] == 'mp4':
    data = json.loads(res.text)

    for k, v in data.items():
        print(k, ": ", v)
else:
    print(res.text)