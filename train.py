import numpy as np
import pandas as pd
import argparse
import os
import random
import pydicom

from tensorflow import keras
import tensorflow as tf

from sklearn.model_selection import train_test_split
from skimage.transform import resize


class generator(keras.utils.Sequence):
    def __init__(self, folder, filenames, data_frame=None, batch_size=32, image_size=256, shuffle=True, augment=False,
                 predict=False):
        self.folder = folder
        self.filenames = filenames
        self.data_frame = data_frame
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.predict = predict
        self.on_epoch_end()

    def __load__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename + '.dcm')).pixel_array
        # create empty mask
        msk = np.zeros(img.shape)

        target = False
        # if image contains pneumonia
        if filename in self.data_frame[self.data_frame['Target'] == 1].index:
            target = True
            # loop through pneumonia
            for location in self.data_frame.values[self.data_frame.index == filename]:
                # add 1's at the location of the pneumonia
                x, y, w, h, _ = location
                msk[y:y + h, x:x + w] = 1
        # resize both image and mask
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        msk = resize(msk, (self.image_size, self.image_size), mode='reflect')  # > 0.5
        # if augment then horizontal flip half the time
        if target and self.augment and random.random() > 0.5:
            img = np.fliplr(img)
            msk = np.fliplr(msk)

        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        msk = np.expand_dims(msk, -1)
        return img, msk

    def __loadpredict__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename + '.dcm')).pixel_array
        # resize image
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        return img

    def __getitem__(self, index):
        # select batch
        filenames = self.filenames[index * self.batch_size:(index + 1) * self.batch_size]
        # predict mode: return images and filenames
        if self.predict:
            # load files
            imgs = [self.__loadpredict__(filename) for filename in filenames]
            # create numpy batch
            imgs = np.array(imgs)
            return imgs, filenames
        # train mode: return images and masks
        else:
            # load files
            items = [self.__load__(filename) for filename in filenames]
            # unzip images and masks
            imgs, msks = zip(*items)
            # create numpy batch
            imgs = np.array(imgs)
            msks = np.array(msks)
            return imgs, msks

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.filenames)

    def __len__(self):
        if self.predict:
            # return everything
            return int(np.ceil(len(self.filenames) / self.batch_size))
        else:
            # return full batches only
            return int(len(self.filenames) / self.batch_size)


def create_downsample(channels, inputs):
    x = keras.layers.BatchNormalization(momentum=0.9)(inputs)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 1, padding='same', use_bias=False)(x)
    x = keras.layers.MaxPool2D(2)(x)
    return x


def create_resblock(channels, inputs):
    x = keras.layers.BatchNormalization(momentum=0.9)(inputs)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    return keras.layers.add([x, inputs])


def create_network(input_size, channels, n_blocks=2, depth=4):
    # input
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(inputs)
    # residual blocks
    for d in range(depth):
        channels = channels * 2
        x = create_downsample(channels, x)
        for b in range(n_blocks):
            x = create_resblock(channels, x)
    # output
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    outputs = keras.layers.UpSampling2D(2 ** depth)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# define iou or jaccard loss function
def iou_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
    return 1 - score


# combine bce loss and iou loss
def iou_bce_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)


# mean iou as a metric
def mean_iou(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))


# cosine learning rate annealing
def cosine_annealing(x):
    lr = 0.001
    epochs = 25
    return lr * (np.cos(np.pi * x / epochs) + 1.) / 2


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--folder_image_path', type=str, help='Path to the folder containing training images')
    parser.add_argument('--class_info_path', type=str, help='Path to the class info CSV file')
    parser.add_argument('--train_labels_path', type=str, help='Path to the training labels CSV file')
    parser.add_argument('--result_model_path', type=str, help='Path to save the trained model weights')

    args = parser.parse_args()

    FOLDER_IMAGE_PATH = args.folder_image_path
    CLASS_INFO_PATH = args.class_info_path
    TRAIN_LABELS_PATH = args.train_labels_path
    RESULT_MODEL_PATH = args.result_model_path

    RANDOM_SEED = 42
    VALID_SIZE = 0.1
    random.seed(RANDOM_SEED)

    # PREPARE DATA
    # Info for images and their labels
    df_info = pd.read_csv(CLASS_INFO_PATH).drop_duplicates().set_index('patientId')
    df_labels = pd.read_csv(TRAIN_LABELS_PATH).set_index('patientId').fillna(0).astype(np.int64)

    # Stratify by "class" because of 3 classes
    images_train, images_valid = train_test_split(df_info.index.values,
                                                  test_size=VALID_SIZE,
                                                  stratify=df_info['class'],
                                                  random_state=RANDOM_SEED)

    print(f"Train shape: {images_train.shape}\nValid shape: {images_valid.shape}")

    # TRAIN MODEL
    # create network and compiler
    model = create_network(input_size=256, channels=32, n_blocks=2, depth=4)
    model.compile(optimizer='adam',
                  loss=iou_bce_loss,
                  metrics=['accuracy', mean_iou])

    learning_rate = tf.keras.callbacks.LearningRateScheduler(cosine_annealing)

    # create train and validation generators
    train_gen = generator(FOLDER_IMAGE_PATH, images_train, df_labels, batch_size=32, image_size=256, shuffle=True,
                          augment=True, predict=False)
    valid_gen = generator(FOLDER_IMAGE_PATH, images_valid, df_labels, batch_size=32, image_size=256, shuffle=False,
                          predict=False)

    # fit model
    history = model.fit(train_gen, validation_data=valid_gen, callbacks=[learning_rate], epochs=3)
    # Save model weights
    model.save_weights(RESULT_MODEL_PATH)
