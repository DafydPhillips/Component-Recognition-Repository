import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dense, Flatten, Conv2D
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds
# import matplotlib as plt
import numpy as np
import cv2
import pathlib
import Utilities_CNN as cnn

# handles webcam feed and predictions made on the images received.
class ComputerVision:
    # initialise image source & prepare the CNN model
    def __init__(self):
        # use webcam in port 0 - if none, try port 1
        self.vision = cv2.VideoCapture(1)
        if not self.vision:
            self.vision = cv2.VideoCapture(0)
        # prepare neural network with a dataset
        self.recog = cnn.NeuralNetwork()
        self.recog.prepare_model()

    # make a prediction on the image received
    def get_prediction(self, img):
        print(img.shape)

        bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh_adpt = cv2.adaptiveThreshold(bw, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 225, 2)

        ret = self.recog.detect(thresh_adpt)

        # display image used for recognition
        pred_img = cv2.resize(thresh_adpt, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Input Image', thresh_adpt)

        return ret

    def get_cam(self):
        # get image from a webcam
        ret, frame = self.vision.read()
        cv2.imshow('frame', frame)

        quit = False
        predict = False

        # either quit if q key is pressed
        # or get a prediction on the current frame if p key is pressed
        if cv2.waitKey(1) == ord('q'):
            self.vision.release()
            cv2.destroyAllWindows()
            quit = True
        if cv2.waitKey(1) == ord('p'):
            predict = self.get_prediction(frame)

        return [quit, predict]

    # receive images from a webcam to make predictions on
    def get_feed(self):
        # while program has not exited
        # prompt the user to either quit the program or make a prediction on the current frame
        print("Enter [q] to quit or [p] to make a prediction")

        # receive image feed until the user quits
        while True:
            # get image from a webcam
            ret, frame = self.vision.read()
            cv2.imshow('frame', frame)

            # either quit if q key is pressed
            # or get a prediction on the current frame if p key is pressed
            if cv2.waitKey(1) == ord('q'):
                break
            if cv2.waitKey(1) == ord('p'):
                self.get_prediction(frame)

        # when the loop ends, release the webcam and close the image window
        self.vision.release()
        cv2.destroyAllWindows()

