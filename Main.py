#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

from deep_sort.detection import Detection as ddet

# EMOTION IMPORTS UNDER HERE

import cv2
import datetime
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import csv

warnings.filterwarnings('ignore')


def main(yolo):
    t = datetime.datetime.now().replace(microsecond=0).isoformat()
    graphInputs = ['6', '8', 'NEUTRAL', '%s' % t, 'MALE', '2']
    with open(r'templates/test2.csv', 'a') as f:

        writer = csv.writer(f)
        writer.writerow(graphInputs)

    # parameters for loading data and images
    emotion_model_path = './models/emotion_model.hdf5'
    emotion_labels = get_labels('fer2013')

    # hyper-parameters for bounding boxes shape
    frame_window = 10
    emotion_offsets = (20, 40)

    # loading models
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    emotion_classifier = load_model(emotion_model_path)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # starting lists for calculating modes
    emotion_window = []

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # deep_sort

    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = False
    resetCounter = 0
    amountOfFramesPerScan = 10
    peopleInFrameList = []
    # video_capture = cv2.VideoCapture('demo/dinner.mp4')
    # video_capture = cv2.VideoCapture('demo/MOT1712.mp4')
    video_capture = cv2.VideoCapture(0)

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break;
        #   -------------------START  EMOTION CODE

        # cv2.imshow('window_frame', frame) SHOWS EMOTION FRAME SEPERATE

        #  --------------------------------   END EMOTION CODE

        t1 = time.time()
        currentPeopleInFrame = 0
        image = Image.fromarray(frame)
        boxs = yolo.detect_image(image)
        # print("box_num",len(boxs))
        features = encoder(frame, boxs)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Imagelist is a list of all the images within the tracked bounding boxes of our tracker.
        imageList = []
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update > 1:
                continue
            # Gets the location of the BBOx coordinates within the tracker.
            bbox = track.to_tlbr()
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            # cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
            currentPeopleInFrame += 1
            # print(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

            numpArr = np.array(frame[int((bbox[1])):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])])
            imageList.append(numpArr)
            # cv2.destroyAllWindows()
        i = 0

        for item in (imageList):
            i += 1
            gray_image = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)

            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                                                  minSize=(0, 0), flags=cv2.CASCADE_SCALE_IMAGE)
            graphInputs[0] = track.track_id
            for face_coordinates in faces:

                x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                gray_face = gray_image[y1:y2, x1:x2]
                try:
                    gray_face = cv2.resize(gray_face, (emotion_target_size))
                except:
                    continue

                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_prediction = emotion_classifier.predict(gray_face)
                emotion_probability = np.max(emotion_prediction)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion_text = emotion_labels[emotion_label_arg]
                emotion_window.append(emotion_text)
                if len(emotion_window) > frame_window:
                    emotion_window.pop(0)
                try:
                    emotion_mode = mode(emotion_window)
                except:
                    continue

                if emotion_text == 'angry':
                    color = emotion_probability * np.asarray((255, 0, 0))
                    print("angry", i)
                    graphInputs[2] = "ANGRY"
                elif emotion_text == 'sad':
                    color = emotion_probability * np.asarray((0, 0, 255))
                    print("sad", i)
                    graphInputs[2] = "SAD"
                elif emotion_text == 'happy':
                    color = emotion_probability * np.asarray((255, 255, 0))
                    print("happy", i)
                    graphInputs[2] = "HAPPY"
                elif emotion_text == 'surprise':
                    color = emotion_probability * np.asarray((0, 255, 255))
                    print("surprise", i)
                    graphInputs[2] = "SURPRISED"
                else:
                    color = emotion_probability * np.asarray((0, 255, 0))
                    print("neutral", i)
                    graphInputs[2] = "NEUTRAL"
                color = color.astype(int)
                color = color.tolist()

                # -------------------------------------

                draw_bounding_box(face_coordinates, rgb_image, color)
                draw_text(face_coordinates, rgb_image, emotion_mode,
                          color, 0, -45, 1, 1)

        cv2.imshow('FilteredImage', frame)
        if resetCounter >= amountOfFramesPerScan:
            peopleInFrameList.append(currentPeopleInFrame)
            print("Total amount of people %d" % (currentPeopleInFrame))

            # Print
            # for x in range(len(peopleInFrameList)):
            #     print("listie  %d" % (peopleInFrameList[x]))

            print(peopleInFrameList)
            resetCounter = 0
        else:
            resetCounter += 1
        print("Geen print of add deze keer %d" % (resetCounter))

        for det in detections:
            bbox = det.to_tlbr()

            currentPeopleInFrame += 1
            # Make
            numpArr = np.array(frame[int((bbox[1])):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])])
            imageList.append(numpArr)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        # cv2.destroyAllWindows()
        i = 0
        for item in (imageList):
            i += 1
            gray_image = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)

            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                                                  minSize=(0, 0), flags=cv2.CASCADE_SCALE_IMAGE)

            for face_coordinates in faces:
                draw_bounding_box(face_coordinates, rgb_image, color)
                draw_text(face_coordinates, rgb_image, emotion_mode,
                          color, 0, -45, 1, 1)

        cv2.imshow('FilteredImage', frame)
        if resetCounter >= amountOfFramesPerScan:
            peopleInFrameList.append(currentPeopleInFrame)
            print("Total amount of people %d" % (currentPeopleInFrame))

            # for x in range(len(peopleInFrameList)):
            #     print("listie  %d" % (peopleInFrameList[x]))
            print(peopleInFrameList)
            resetCounter = 0
        else:
            resetCounter += 1
        print("Geen print of add deze keer %d" % (resetCounter))

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index) + ' ')
            if len(boxs) != 0:
                for i in range(0, len(boxs)):
                    list_file.write(
                        str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
            list_file.write('\n')

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %f" % (fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())
