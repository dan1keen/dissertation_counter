import tensorflow as tf
import csv
import cv2
import numpy as np
import mysql.connector
from mysql.connector import Error
from mysql.connector import errorcode
from utils import visualization_utils as vis_util
from datetime import datetime

# Variables
total_passed_vehicle = 0  # using it to count vehicles


def saveVehicle(object_name, count, date, output_name):
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='python_items',
                                             user='root',
                                             password='root')
        cursor = connection.cursor()
        sql_insert_query = """ INSERT INTO `vehicle`
                              (`object`, `count`, `date`, `name_text`) VALUES (%s,%s,%s,%s)"""
        insert_tuple = (object_name, count, date, output_name)
        result = cursor.execute(sql_insert_query, insert_tuple)
        connection.commit()
        print("Record inserted successfully into python_users table")
    except mysql.connector.Error as error:
        connection.rollback()  # rollback if any exception occured
        print("Failed inserting record into items table {}".format(error))
    finally:
        # closing database connection.
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")


def savePedestrian(object_name, count, date, output_name):
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='python_items',
                                             user='root',
                                             password='root')
        cursor = connection.cursor()
        sql_insert_query = """ INSERT INTO `pedestrian`
                              (`object`, `count`, `date`, `name_text`) VALUES (%s,%s,%s,%s)"""
        insert_tuple = (object_name, count, date, output_name)
        result = cursor.execute(sql_insert_query, insert_tuple)
        connection.commit()
        print("Record inserted successfully into python_users table")
    except mysql.connector.Error as error:
        connection.rollback()  # rollback if any exception occured
        print("Failed inserting record into items table {}".format(error))
    finally:
        # closing database connection.
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")


def cumulative_object_counting_x_axis(input_video, detection_graph, category_index, is_color_recognition_enabled,
                                      targeted_objects, fps, roi, deviation):
    total_passed_vehicle = 0

    # initialize .csv
    with open('object_counting_report.csv', 'w') as f:
        writer = csv.writer(f)
        csv_line = "Object Type, Object Color, Object Movement Direction, Object Speed (km/h)"
        writer.writerows([csv_line.split(',')])

    # input video
    cap = cv2.VideoCapture(input_video)
    if cap.isOpened():
        # get cap property
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter('the_object_x_axis.avi', fourcc, fps, (width, height))

    total_passed_vehicle = 0
    speed = "waiting..."
    direction = "waiting..."
    size = "waiting..."
    color = "waiting..."
    counting_mode = "..."
    width_heigh_taken = True
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while (cap.isOpened()):
                ret, frame = cap.read()

                if not ret:
                    print("end of the video file...")
                    break

                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.
                counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array_x_axis(cap.get(1),
                                                                                                             input_frame,
                                                                                                             1,
                                                                                                             is_color_recognition_enabled,
                                                                                                             np.squeeze(
                                                                                                                 boxes),
                                                                                                             np.squeeze(
                                                                                                                 classes).astype(
                                                                                                                 np.int32),
                                                                                                             np.squeeze(
                                                                                                                 scores),
                                                                                                             category_index,
                                                                                                             targeted_objects=targeted_objects,
                                                                                                             x_reference=roi,
                                                                                                             deviation=deviation,
                                                                                                             use_normalized_coordinates=True,
                                                                                                             line_thickness=4)

                # when the vehicle passed over line and counted, make the color of ROI line green
                if counter == 1:
                    cv2.line(input_frame, (roi, 0), (roi, height), (0, 0xFF, 0), 5)
                else:
                    cv2.line(input_frame, (roi, 0), (roi, height), (0, 0, 0xFF), 5)

                total_passed_vehicle = total_passed_vehicle + counter
                time = datetime.now().strftime('%Y-%m-%d %H:%M')

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Detected Pedestrians: ' + str(total_passed_vehicle),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                )

                cv2.putText(
                    input_frame,
                    'ROI Line',
                    (545, roi - 10),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,
                    cv2.LINE_AA,
                )

                output_movie.write(input_frame)
                print("writing frame")
                cv2.imshow('object counting', input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                '''if(csv_line != "not_available"):
                        with open('traffic_measurement.csv', 'a') as f:
                                writer = csv.writer(f)                          
                                size, direction = csv_line.split(',')                                             
                                writer.writerows([csv_line.split(',')])         '''
            print(total_passed_vehicle)
            if(targeted_objects=="person"):
                compare_last_pedestrian("pedestrian", total_passed_vehicle, time, "the_object_y_axis.avi")
            elif(targeted_objects=="car"):
                compare_last_vehicle("vehicle", total_passed_vehicle, time, "the_object_y_axis.avi")
            cap.release()
            cv2.destroyAllWindows()


def compare_last_vehicle(obj, count, date, output_name):
    try:
        mySQLConnection = mysql.connector.connect(host='localhost',
                                                  database='python_items',
                                                  user='root',
                                                  password='root')
        cursor = mySQLConnection.cursor()
        sql_select_query = """select * from vehicle order by id desc limit 1"""
        cursor.execute(sql_select_query)
        record = cursor.fetchone()
        print(record)
        if (record != None):

            date_compare = record[2]
            print("Just hour w/o minute = ", date_compare[11:-3])
            print("Just year/month/day = ", date_compare[0:-6])
            if (date_compare[11:-3] == date[11:-3] and date_compare[0:-6] == date[0:-6]):
                print("Counted objects sum is the same!!")
            else:
                saveVehicle(obj, count, date, output_name)
        else:
            saveVehicle(obj, count, date, output_name)
    except mysql.connector.Error as error:
        print("Failed to get record from database: {}".format(error))
    finally:
        # closing database connection.
        if (mySQLConnection.is_connected()):
            cursor.close()
            mySQLConnection.close()
            print("connection is closed")


def compare_last_pedestrian(obj, count, date, output_name):
    try:
        mySQLConnection = mysql.connector.connect(host='localhost',
                                                  database='python_items',
                                                  user='root',
                                                  password='root')
        cursor = mySQLConnection.cursor()
        sql_select_query = """select * from pedestrian order by id desc limit 1"""
        cursor.execute(sql_select_query)
        record = cursor.fetchone()
        print(record)
        if (record != None):

            date_compare = record[2]
            print("Just hour w/o minute = ", date_compare[11:-3])
            print("Just year/month/day = ", date_compare[0:-6])
            if (date_compare[11:-3] == date[11:-3] and date_compare[0:-6] == date[0:-6]):
                print("Counted objects sum is the same!!")
            else:
                savePedestrian(obj, count, date, output_name)
        else:
            savePedestrian(obj, count, date, output_name)
    except mysql.connector.Error as error:
        print("Failed to get record from database: {}".format(error))
    finally:
        # closing database connection.
        if (mySQLConnection.is_connected()):
            cursor.close()
            mySQLConnection.close()
            print("connection is closed")


def rewritePedestrian(obj, count, date, output_name):
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='python_items',
                                             user='root',
                                             password='root')
        cursor = connection.cursor()
        sql_insert_query = """ UPDATE pedestrian SET object = %s, count = %s, date = %s, name_text = %s WHERE id IN 
        (select max(id) from (select * from pedestrian) AS pd)"""
        insert_tuple = (obj, count, date, output_name)
        result = cursor.execute(sql_insert_query, insert_tuple)
        connection.commit()
        print("Record inserted successfully into python_users table")
    except mysql.connector.Error as error:
        connection.rollback()  # rollback if any exception occured
        print("Failed inserting record into items table {}".format(error))
    finally:
        # closing database connection.
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")


def compare_max_pedestrian(obj, count, date, output_name):
    try:
        mySQLConnection = mysql.connector.connect(host='localhost',
                                                  database='python_items',
                                                  user='root',
                                                  password='root')
        cursor = mySQLConnection.cursor()
        sql_select_query = """select * from pedestrian order by id desc limit 1"""
        cursor.execute(sql_select_query)
        record = cursor.fetchone()
        print(record)
        if (record != None):

            date_compare = record[2]
            print("Just hour w/o minute = ", date_compare[11:-3])
            print("Just year/month/day = ", date_compare[0:-6])
            if (date_compare[11:-3] == date[11:-3] and date_compare[0:-6] == date[0:-6]):
                if (int(record[1]) < count):
                    rewritePedestrian(obj, count, date, output_name)
                print("Counted objects sum is the same!!")
            else:

                savePedestrian(obj, count, date, output_name)
        else:
            savePedestrian(obj, count, date, output_name)
    except mysql.connector.Error as error:
        print("Failed to get record from database: {}".format(error))
    finally:
        # closing database connection.
        if (mySQLConnection.is_connected()):
            cursor.close()
            mySQLConnection.close()
            print("connection is closed")


def cumulative_object_counting_y_axis(input_video, detection_graph, category_index, is_color_recognition_enabled,
                                      targeted_object, fps, roi, deviation):
    # initialize .csv
    with open('traffic_measurement.csv', 'w') as f:
        writer = csv.writer(f)
        csv_line = \
            'Vehicle Type/Size, Vehicle Color, Vehicle Movement Direction, Vehicle Speed (km/h)'
        writer.writerows([csv_line.split(',')])

    # input video
    cap = cv2.VideoCapture(input_video)
    if cap.isOpened():
        # get cap property
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter('the_object_y_axis.avi', fourcc, fps, (width, height))
    total_passed_vehicle = 0
    speed = "waiting..."
    direction = "waiting..."
    size = "waiting..."
    color = "waiting..."
    counting_mode = "..."
    width_heigh_taken = True
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while (cap.isOpened()):
                ret, frame = cap.read()

                if not ret:
                    print("end of the video file...")
                    break
                # else:
                #     cv2.imshow('frame', frame)
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break
                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.
                counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array_y_axis(cap.get(1),
                                                                                                             input_frame,
                                                                                                             2,
                                                                                                             is_color_recognition_enabled,
                                                                                                             np.squeeze(
                                                                                                                 boxes),
                                                                                                             np.squeeze(
                                                                                                                 classes).astype(
                                                                                                                 np.int32),
                                                                                                             np.squeeze(
                                                                                                                 scores),
                                                                                                             category_index,
                                                                                                             targeted_objects=targeted_object,
                                                                                                             y_reference=roi,
                                                                                                             deviation=deviation,
                                                                                                             use_normalized_coordinates=True,
                                                                                                             min_score_thresh=.0,
                                                                                                             line_thickness=4)

                # when the vehicle passed over line and counted, make the color of ROI line green
                if counter == 1:
                    cv2.line(input_frame, (0, roi), (width, roi), (0, 0xFF, 0), 5)
                else:
                    cv2.line(input_frame, (0, roi), (width, roi), (0, 0, 0xFF), 5)

                total_passed_vehicle = total_passed_vehicle + counter

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Detected Vehicles: ' + str(total_passed_vehicle),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                )

                cv2.putText(
                    input_frame,
                    'ROI Line',
                    (545, roi - 10),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    input_frame,
                    'LAST PASSED VEHICLE INFO',
                    (11, 290),
                    font,
                    0.5,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_SIMPLEX,
                )
                cv2.putText(
                    input_frame,
                    '-Movement Direction: ' + direction,
                    (14, 302),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                )
                cv2.putText(
                    input_frame,
                    '-Speed(km/h): ' + speed,
                    (14, 312),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                )
                cv2.putText(
                    input_frame,
                    '-Color: ' + color,
                    (14, 322),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                )
                cv2.putText(
                    input_frame,
                    '-Vehicle Size/Type: ' + size,
                    (14, 332),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                )

                output_movie.write(input_frame)
                print("writing frame")
                cv2.imshow('object counting', input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if csv_line != 'not_available':
                    with open('traffic_measurement.csv', 'a') as f:
                        writer = csv.writer(f)
                        (size, direction) = \
                            csv_line.split(',')
                        writer.writerows([csv_line.split(',')])
            print(total_passed_vehicle)
            time = datetime.now().strftime('%Y-%m-%d %H:%M')
            date = datetime.now().strftime('%Y-%m-%d')
            if(targeted_object=="car"):
                compare_last_vehicle("vehicle", total_passed_vehicle, time, "the_object_y_axis.avi")
            elif(targeted_object=="person"):
                compare_last_pedestrian("pedestrian", total_passed_vehicle, time, "the_object_y_axis.avi")

            cap.release()
            cv2.destroyAllWindows()


def object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height):
    # initialize .csv
    with open('object_counting_report.csv', 'w') as f:
        writer = csv.writer(f)
        csv_line = "Object Type, Object Color, Object Movement Direction, Object Speed (km/h)"
        writer.writerows([csv_line.split(',')])

    # input video
    cap = cv2.VideoCapture(input_video, cv2.CAP_DSHOW)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))
    total_passed_vehicle = 0
    speed = "waiting..."
    direction = "waiting..."
    size = "waiting..."
    color = "waiting..."
    counting_mode = "..."
    width_heigh_taken = True
    height = 0
    width = 0
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while (cap.isOpened()):
                ret, frame = cap.read()
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if not ret:
                    print("end of the video file...")
                    break

                input_frame = frame
                # else:
                #     cv2.imshow('frame', frame)
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.
                counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                      input_frame,
                                                                                                      1,
                                                                                                      is_color_recognition_enabled,
                                                                                                      np.squeeze(boxes),
                                                                                                      np.squeeze(
                                                                                                          classes).astype(
                                                                                                          np.int32),
                                                                                                      np.squeeze(
                                                                                                          scores),
                                                                                                      category_index,
                                                                                                      use_normalized_coordinates=True,
                                                                                                      line_thickness=4)
                if (len(counting_mode) == 0):
                    cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)
                else:
                    cv2.putText(input_frame, counting_mode, (10, 35), font, 0.8, (0, 255, 255), 2,
                                cv2.FONT_HERSHEY_SIMPLEX)

                output_movie.write(input_frame)
                print("writing frame")

                cv2.imshow('object counting', input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if (csv_line != "not_available"):
                    with open('traffic_measurement.csv', 'a') as f:
                        writer = csv.writer(f)
                        size, direction = csv_line.split(',')
                        writer.writerows([csv_line.split(',')])

            cap.release()
            cv2.destroyAllWindows()


def targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled,
                             targeted_object, fps):
    # initialize .csv
    with open('object_counting_report.csv', 'w') as f:
        writer = csv.writer(f)
        csv_line = "Object Type, Object Color, Object Movement Direction, Object Speed (km/h)"
        writer.writerows([csv_line.split(',')])

    # input video
    # cap = cv2.VideoCapture(input_video) # recorded video file mode
    cap = cv2.VideoCapture(0) # WebCamera mode
    if cap.isOpened():
        # get cap property
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))
    total_passed_vehicle = 0
    speed = "waiting..."
    direction = "waiting..."
    size = "waiting..."
    color = "waiting..."
    the_result = "..."
    width_heigh_taken = True
    height = 0
    width = 0
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while (cap.isOpened()):
                ret, frame = cap.read()

                if not ret:
                    print("end of the video file...")
                    break

                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.
                counter, csv_line, the_result = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                   input_frame,
                                                                                                   1,
                                                                                                   is_color_recognition_enabled,
                                                                                                   np.squeeze(boxes),
                                                                                                   np.squeeze(
                                                                                                       classes).astype(
                                                                                                       np.int32),
                                                                                                   np.squeeze(scores),
                                                                                                   category_index,
                                                                                                   targeted_objects=targeted_object,
                                                                                                   use_normalized_coordinates=True,
                                                                                                   line_thickness=4)
                if (len(the_result) == 0):
                    cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)
                else:
                    cv2.putText(input_frame, the_result, (10, 35), font, 0.8, (0, 255, 255), 2,
                                cv2.FONT_HERSHEY_SIMPLEX)
                    if (targeted_object == "person"):
                        if (len(the_result) > 12):
                            total_passed_vehicle = int(the_result[11:13])
                        else:
                            total_passed_vehicle = int(the_result[11])

                    elif (targeted_object == "car"):
                        if (len(the_result) > 9):
                            total_passed_vehicle = int(the_result[8:10])
                        else:
                            total_passed_vehicle = int(the_result[8])

                cv2.imshow('object counting', input_frame)

                output_movie.write(input_frame)
                print("writing frame")
                print(total_passed_vehicle)
                time = datetime.now().strftime('%Y-%m-%d %H:%M')

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if (csv_line != "not_available"):
                    with open('traffic_measurement.csv', 'a') as f:
                        writer = csv.writer(f)
                        size, direction = csv_line.split(',')
                        writer.writerows([csv_line.split(',')])
                if (targeted_object == "person"):
                    compare_max_pedestrian("pedestrian", total_passed_vehicle, time, "the_pedestrian_output.avi")
                elif (targeted_object == "car"):
                    print("Will be programmed soon!!")
            cap.release()
            cv2.destroyAllWindows()


def single_image_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width,
                                 height):
    total_passed_vehicle = 0
    speed = "waiting..."
    direction = "waiting..."
    size = "waiting..."
    color = "waiting..."
    counting_mode = "..."
    width_heigh_taken = True
    height = 0
    width = 0
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    input_frame = cv2.imread(input_video)

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(input_frame, axis=0)

    # Actual detection.
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # insert information text to video frame
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Visualization of the results of a detection.
    counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_single_image_array(1, input_frame,
                                                                                                 1,
                                                                                                 is_color_recognition_enabled,
                                                                                                 np.squeeze(boxes),
                                                                                                 np.squeeze(
                                                                                                     classes).astype(
                                                                                                     np.int32),
                                                                                                 np.squeeze(scores),
                                                                                                 category_index,
                                                                                                 use_normalized_coordinates=True,
                                                                                                 line_thickness=4)
    if (len(counting_mode) == 0):
        cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)
    else:
        cv2.putText(input_frame, counting_mode, (10, 35), font, 0.8, (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)

    cv2.imshow('tensorflow_object counting_api', input_frame)
    cv2.waitKey(0)

    return counting_mode

