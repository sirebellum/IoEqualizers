from __future__ import print_function
from PIL import Image
import tensorflow as tf
import numpy as np
import base64
import json
import requests

# Server data
server = 'localhost:8501' # Rest API port, grpc is 8500
model_name = 'feedback'
signature_name = 'predict_class'
url = "http://"+server+"/v1/models/"+model_name+":predict"

# Function to create instance json for multiple images
def create_json(images, signature_name):

    # Assume size is the same
    height = images[0].shape[0]
    width = images[0].shape[1]
    # Flatten for transmission
    images = [image.flatten().tolist() for image in images]

    # Top level request Json with list of instances
    request_dict = {"instances": [], "signature_name": signature_name}
    
    # fill in the request dictionary with the necessary data
    for image in images:
        request_dict["instances"].append( {"image": image, \
                                           "height": height, \
                                           "width" : width} )
    # Jsonize
    json_request = json.dumps(request_dict)
    
    return json_request

# Feedback data
import data_processing.audio_processing as ap
dataset = ap.nsynth("nsynth/nsynth-test", fb=True)

# Get data
batch = dataset.returnInstance(10)
images = [ ap.plotSpectrumBW(image) for image in batch['fft'] ]
images = np.asarray(images, dtype=np.float32)
images *= 1/255.0

# Turn into json request
json_request = create_json(images, signature_name)

# Get preidctions
output = requests.post(url, data=json_request)
predictions = output.json()['predictions']

print(predictions, batch['fb'])