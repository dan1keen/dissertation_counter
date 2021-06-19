
# Imports
import tensorflow as tf
from tkinter import Tk
from tkinter import *
from tkinter.filedialog import askopenfilename
import numpy as np
import cv2
# Object detection imports
from utils import backbone
from api import object_counting_api

from kalman_tracker import detector
from kalman_tracker import main

import os

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

window = Tk()
window.title("File selector")
window.geometry('225x200')
mainframe = Frame(window)
mainframe.grid(column=0,row=0, sticky=(N,W,E,S) )
mainframe.columnconfigure(0, weight = 1)
mainframe.rowconfigure(0, weight = 1)

tkvarRoot = StringVar(window)
tkvar = StringVar(window)
tkvarObj = StringVar(window)
tkvarAxis = StringVar(window)
types = {'Tensorflow with Kalman Filter', 'Tensorflow Object Detection API'}
choices = {'faster_rcnn_inception_v2_coco_2018_01_28', 'ssd_mobilenet_v1_coco_2018_01_28', 'ssdlite_mobilenet_v2_coco_2018_05_09',
           'faster_rcnn_resnet101_coco_2018_01_28'}
objChoices = {'vehicle', 'pedestrian', 'real time pedestrian','real time vehicle'}
axChoices = {'by x axis', 'by y axis'}
tkvar.set('ssd_mobilenet_v1_coco_2018_01_28')  # set the default option
tkvarRoot.set('Choose')  # set the default option
tkvarObj.set('Choose an object')
tkvarAxis.set('Choose an axis')
popupMenuRoot = OptionMenu(mainframe, tkvarRoot, *types)
popupMenuRoot.grid(row=0, column=0)



def vehicle_clicked():
  Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
  filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
  input_video = filename
  print(input_video)

  # By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
  detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28')
  targeted_objects = "car"
  fps = 24  # change it with your input video fps
  is_color_recognition_enabled = 0  # set it to 1 for enabling the color prediction for the detected objects
  roi = 200  # roi line position

  deviation = 3  # the constant that represents the object counting area
  if(tkvarAxis.get()=="by y axis"):
    object_counting_api.cumulative_object_counting_y_axis(input_video, detection_graph, category_index,
                                                          is_color_recognition_enabled,targeted_objects, fps, 200,
                                                          deviation)  # counting all the objects
    btn = Button(window, text="Show Result", command=result)
    btn.grid(column=0, row=7)

  elif(tkvarAxis.get()=="by x axis"):
    object_counting_api.cumulative_object_counting_x_axis(input_video, detection_graph, category_index,
                                                          is_color_recognition_enabled, targeted_objects, fps, 385,
                                                          deviation)  # counting all the objects
    btn = Button(window, text="Show Result", command=result_x)
    btn.grid(column=0, row=7)

def pedestrian_clicked():
  Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
  filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
  input_video = filename
  print(input_video)

  # By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
  detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28')
  targeted_objects = "person"
  fps = 20  # change it with your input video fps
  is_color_recognition_enabled = 0  # set it to 1 for enabling the color prediction for the detected objects
  roi = 385  # roi line position
  deviation = 1  # the constant that represents the object counting area

  if (tkvarAxis.get() == "by y axis"):
    object_counting_api.cumulative_object_counting_y_axis(input_video, detection_graph, category_index,
                                                          is_color_recognition_enabled, targeted_objects, fps, 200,
                                                          deviation)  # counting all the objects

    btn = Button(window, text="Show Result", command=result)
    btn.grid(column=0, row=7)
  elif (tkvarAxis.get() == "by x axis"):
    object_counting_api.cumulative_object_counting_x_axis(input_video, detection_graph, category_index,
                                                          is_color_recognition_enabled, targeted_objects, fps, 385,
                                                          deviation)  # counting all the objects

    btn = Button(window, text="Show Result", command=result_x)
    btn.grid(column=0, row=7)

def real_pedestrian_clicked():
  Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
  filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
  input_video = filename
  # input_video = 0
  print(input_video)

  # By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
  detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28')
  targeted_objects = "person"
  fps = 24  # change it with your input video fps
  is_color_recognition_enabled = 0  # set it to 1 for enabling the color prediction for the detected objects
  roi = 200  # roi line position
  deviation = 3  # the constant that represents the object counting area

  object_counting_api.targeted_object_counting(input_video, detection_graph, category_index,
                                                        is_color_recognition_enabled,targeted_objects, fps)  # counting all the objects

  btn = Button(window, text="Show Result", command=result_real)
  btn.grid(column=0, row=7)

def real_vehicle_clicked():
  Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
  filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
  input_video = filename
  # input_video = 0
  print(input_video)

  # By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
  detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28')
  targeted_objects = "car"
  fps = 24  # change it with your input video fps
  is_color_recognition_enabled = 0  # set it to 1 for enabling the color prediction for the detected objects
  roi = 200  # roi line position
  deviation = 3  # the constant that represents the object counting area

  object_counting_api.targeted_object_counting(input_video, detection_graph, category_index,
                                                        is_color_recognition_enabled,targeted_objects, fps)  # counting all the objects

  btn = Button(window, text="Show Result", command=result_real)
  btn.grid(column=0, row=7)

def result():
  cap = cv2.VideoCapture('./the_object_y_axis.avi')

  while (cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
      cv2.imshow('frame', frame)
      if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    else:
      break
  cap.release()
  cv2.destroyAllWindows()
def result_x():
  cap = cv2.VideoCapture('./the_object_x_axis.avi')

  while (cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
      cv2.imshow('frame', frame)
      if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    else:
      break
  cap.release()
  cv2.destroyAllWindows()
def result_real():
  cap = cv2.VideoCapture('./the_output.avi')

  while (cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
      cv2.imshow('frame', frame)
      if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    else:
      break
  cap.release()
  cv2.destroyAllWindows()

def change_dropdown_root(*args):
  if (tkvarRoot.get() == 'Tensorflow with Kalman Filter'):
    cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('input_images_and_videos/test2.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 8.0, (640, 480))
    det = detector.PersonDetector()
    roi = 200
    counter = []
    while (True):

      ret, img = cap.read()
      # print(img)

      np.asarray(img)
      font = cv2.FONT_HERSHEY_SIMPLEX
      # trackers_count = pipeline(img)[1]

      new_img = main.pipeline(img, det)
      out.write(new_img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()
  else:
    os.chdir('D:/Downloads/Диссертация/tensorflow_object_counting_api-master/tensorflow_object_counting_api-master/')
    print(format(os.getcwd()))
    # popupMenu = OptionMenu(mainframe, tkvar, *choices)
    # Label(mainframe, text="Choose a dataset model").grid(row=1, column=0)
    # popupMenu.grid(row=2, column=0)
    popupMenu2 = OptionMenu(mainframe, tkvarObj, *objChoices)
    popupMenu2.grid(row=3, column=0)

tkvarRoot.trace('w', change_dropdown_root)
# on change dropdown value
def change_dropdown(*args):
  print(tkvar.get())

# link function to change dropdown
tkvar.trace('w', change_dropdown)

def change_dropdown2(*args):
  print(tkvarObj.get())
  popupMenu3 = OptionMenu(mainframe, tkvarAxis, *axChoices)
  if(tkvarObj.get()=='vehicle'):
    btn = Button(window, text="Open", command=vehicle_clicked)
    btn.grid(column=0, row=6)

    popupMenu3.grid(row=4, column=0)
  elif(tkvarObj.get()=='pedestrian'):
    btn = Button(window, text="Open", command=pedestrian_clicked)
    btn.grid(column=0, row=6)

    popupMenu3.grid(row=4, column=0)
  elif(tkvarObj.get()=='real time pedestrian'):
    btn = Button(window, text="Open", command=real_pedestrian_clicked)
    btn.grid(column=0, row=6)


  elif(tkvarObj.get()=='real time vehicle'):
    btn = Button(window, text="Open", command=real_vehicle_clicked)
    btn.grid(column=0, row=6)


  else:
    lbl = Label(window, text="Required!!")
    lbl.grid(column=0, row=6)
  lbl = Label(window, text="Choose a video")
  lbl.grid(column=0, row=5)
# link function to change dropdown
tkvarObj.trace('w', change_dropdown2)

def change_dropdown3(*args):
  print(tkvarAxis.get())



# link function to change dropdown
tkvarAxis.trace('w', change_dropdown3)

window.mainloop()



