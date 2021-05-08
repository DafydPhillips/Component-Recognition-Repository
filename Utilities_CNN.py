import tensorflow as tf
from tensorflow.keras import layers, models
# from tensorflow.keras.layers import Dense, Flatten, Conv2D
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import tensorflow_datasets as tfds
# import matplotlib as plt
import numpy as np
import cv2
import pathlib
import os
import random

# prepare dataset directory for testing
# only this part of the program is aware of the dataset - the rest receives data from the NN
data_dir = "./Shape_Data"
data_dir = pathlib.Path(data_dir)


# custom domain randomisation layer - adds random noise to an image
class AddNoise(tf.keras.layers.Layer):
    # get values for use in randomising noise
    def __init__(self, mean=0.0, dev=0.1, name=None, **kwargs):
        self.mean = mean
        self.dev = dev
        # initialise superclass Layer
        super(AddNoise, self).__init__(name, **kwargs)

    # adds Gaussian noise to an image for preprocessing
    def add_noise_to_img(self, img):
        # create noise
        noise = tf.random.normal(shape=tf.shape(img), mean=self.mean, stddev=self.dev, dtype=tf.float32)
        # add noise to the image through tensor addition
        img_noise = tf.add(img, noise)
        # return noisy image
        return img_noise

    # create a random chance for an image to have noise added to it
    def randomise_noise(self, img):
        chance_of_noise = random.random()

        if chance_of_noise > 0.8:
            # add noise to the image
            img = self.add_noise_to_img(img)
        # return noisy image
        return img

    # run method to add noise when called and return tensor output
    def call(self, input):
        output = self.randomise_noise(input)
        return output


# class that handles neural network values and functions
class NeuralNetwork:

    # initialises network model and training dataset
    # dataset is built from local directory
    # image width and height is normalised
    def __init__(self, directory=data_dir, img_width=100, img_height=100):
        # prepare model folder for loading/saving model weights
        self.model_folder = "./data/test_model_5"
        self.ckpt_path = str(self.model_folder) + "/conv.ckpt"
        self.ckpt_dir = os.path.dirname(self.ckpt_path)

        # get a list of class names for future use
        self.class_names = os.listdir(directory)

        # initialise size of training batches
        batch_size = 32
        # infer training dataset from the dataset directory
        # 20% of the images are used for validation
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory,
            labels='inferred',
            validation_split=0.2,
            subset="training",
            seed=123,
            batch_size=batch_size,
            follow_links=True
        )

        # get number of classes to prepare number of outputs
        no_of_classes = len(self.train_ds.class_names)

        # infer validation dataset from dataset directory
        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory,
            validation_split=0.2,
            subset="validation",
            seed=123,
            batch_size=batch_size,
            follow_links=True
        )

        # initialise preprocessing layers #
        # allows domain randomisation
        # provides greater consistency between image dimensions
        self.img_width = img_width
        self.img_height = img_height

        # initialise input layer to receive image arrays
        input_layer = tf.keras.Input(shape=(256, 256, 3))

        # create sequential layers to resize and normalise image arrays
        resize_rescale = tf.keras.Sequential([
            layers.experimental.preprocessing.Resizing(self.img_height, self.img_width),
            layers.experimental.preprocessing.Rescaling(1./255)  # standardizes pixels as [0, 1]
        ])

        # create sequential layers to add random preprocessing effects to training data
        domain_randomisation = tf.keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            layers.experimental.preprocessing.RandomRotation(0.2),
            layers.experimental.preprocessing.RandomContrast(0.2),
            layers.experimental.preprocessing.RandomZoom(0.2),
            AddNoise(0.0, 0.1)
        ])

        # initialise model
        self.model = models.Sequential()
        # add preprocessing layers
        self.model.add(input_layer)
        self.model.add(resize_rescale)
        self.model.add(domain_randomisation)
        # create 2 Convolutional layers and 2 Max Pooling layers, one after the other
        # Conv layers convolute images into smaller sections
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        # Pooling layers simplify and compress images further to improve efficiency
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))

        # flatten layers to 1D
        self.model.add(layers.Flatten())
        # connects flattened layer to output
        self.model.add(layers.Dense(64, activation='relu'))
        # create output layer with node equal to the number of classes
        self.model.add(layers.Dense(no_of_classes, activation='softmax'))
        # build model
        self.model.build((None, 32, 32, 3))

        # prepare model to be trained using the Adam optimiser
        # loss uses Crossentropy
        # main metric is model accuracy
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        # view model summary
        self.model.summary()

    # view the loss function and accuracy of the model
    def eval_model(self):
        test_loss, test_acc = self.model.evaluate(self.val_ds, verbose=2)
        print(test_loss)
        print(test_acc)

    # either train or load weights into the model to prepare for use
    def prepare_model(self):
        # if model directory has no stored weights, train the model from scratch
        if not os.listdir(self.model_folder):
            self.train_model()
        else:
            self.load_model()

    # get an entry from the dataset to view
    def get_ds_info(self):
        numpy_img = None
        numpy_label = None
        for img, label in self.train_ds.take(1):
            numpy_img = img.numpy()
            numpy_label = label.numpy()
        return numpy_img, numpy_label

    # train the model for a set number of epochs
    def train_model(self, epoch=50):
        # prepare callback to save weights as epochs are finished
        callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.ckpt_path,
            save_weights_only=True,
            verbose=1
        )

        # fit training and validation dataset to the model for a set number of epochs
        # wieghts are saved to a directory as the model is trained
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epoch,
            callbacks=[callback]

        )

        # get an evaluation of the trained model
        self.eval_model()

    # load and evaluate model weights
    def load_model(self):
        self.model.load_weights(self.ckpt_path)
        self.eval_model()

    # get a set of predictions for a given image
    def detect(self, img):
        # resize and reshape image array to fit model input layer
        img_resize = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        img_array = np.asarray(img_resize)
        # if image is grayscale, add empty colour channels
        if img_array.shape != (256, 256, 3):
            dim = np.zeros((256, 256))
            img_array = np.stack((img_array, dim, dim), axis=2)
        img_input = np.expand_dims(img_array, axis=0)

        print(img_input.shape)

        # store array of predictions
        predictions = self.model.predict(img_input)
        predict_list = []

        # for every class, print its name and the corresponding prediction made
        # predictions are given in order of the dataset
        for predict in range(len(predictions[0])):
            class_name = self.class_names[predict]
            print("\nClass ", class_name, ": ", predictions[0][predict]*100, "% likelihood")
            # build list of ['class_name', 'class_predictions']
            predict_list.append([[class_name], predictions[0][predict]*100])
        print("")
        return predict_list
