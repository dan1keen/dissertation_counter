
# Imports
import tensorflow as tf

# Object detection imports
from utils import backbone
from api import object_counting_api

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

input_video = "./input_images_and_videos/sample_input_image.jpg"

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2017_11_17')

fps = 30 # change it with your input video fps
width = 626 # change it with your input video width
height = 360 # change it with your input vide height
is_color_recognition_enabled = 0

result = object_counting_api.single_image_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height) # targeted objects counting

print (result)
