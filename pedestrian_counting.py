# Imports
import tensorflow as tf

# Object detection imports
from utils import backbone
from api import object_counting_api

if tf.VERSION > '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

input_video = "./input_images_and_videos/pedestrian_survaillance.mp4"

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28')

fps = 30 # change it with your input video fps
width = 626 # change it with your input video width
height = 360 # change it with your input vide height
is_color_recognition_enabled = 0 # set it to 1 for enabling the color prediction for the detected objects
roi = 385 # roi line position
deviation = 1 # the constant that represents the object counting area

object_counting_api.cumulative_object_counting_x_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height, roi, deviation) # counting all the objects
