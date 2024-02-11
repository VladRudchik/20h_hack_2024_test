import numpy as np
import pandas as pd
import argparse
import os
import random
import pydicom
import requests

from tensorflow import keras

from skimage import measure
from skimage.transform import resize


def download_file_from_google_drive(url, destination):
    session = requests.Session()

    response = session.get(url, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


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


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Predict and create a submission CSV for image classification.')
    parser.add_argument('--test_folder_path', type=str, help='Path to the folder containing test images')
    parser.add_argument('--result_csv_path', type=str, default='submission.csv', help='Path to save the submission CSV file')

    args = parser.parse_args()

    test_folder_path = args.test_folder_path
    result_csv_path = args.result_csv_path

    # Load model weights
    destination = 'model_weights.h5'
    url = 'https://drive.google.com/uc?id=1L-Ied1uvzJNR6ZolxmL01OuDyxOXJGEd'
    download_file_from_google_drive(url, destination)

    # Initialize model
    model = create_network(input_size=256, channels=32, n_blocks=2, depth=4)
    model.load_weights('model_weights.h5')

    # load and shuffle filenames
    test_names = os.listdir(test_folder_path)
    test_filenames = []
    for filename in test_names:
        test_filenames.append(filename.split('.')[0])

    print('n test samples:', len(test_filenames))

    # create test generator with predict flag set to True
    test_gen = generator(test_folder_path, test_filenames, None, batch_size=1000, image_size=256, shuffle=False,
                         predict=True)

    # create submission dictionary
    submission_dict = {}
    # loop through testset
    for imgs, filenames in test_gen:
        # predict batch of images
        preds = model.predict(imgs)
        # loop through batch
        for pred, filename in zip(preds, filenames):
            # resize predicted mask
            pred = resize(pred, (1024, 1024), mode='reflect')
            # threshold predicted mask
            comp = pred[:, :, 0] > 0.5
            # apply connected components
            comp = measure.label(comp)
            # apply bounding boxes
            predictionString = ''
            for region in measure.regionprops(comp):
                # retrieve x, y, height and width
                y, x, y2, x2 = region.bbox
                height = y2 - y
                width = x2 - x
                # proxy for confidence score
                conf = np.mean(pred[y:y + height, x:x + width])
                # add to predictionString
                predictionString += str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + ' '
            # add filename and predictionString to dictionary
            filename = filename.split('.')[0]
            submission_dict[filename] = predictionString
        # stop if we've got them all
        if len(submission_dict) >= len(test_filenames):
            break

    # save dictionary as csv file
    sub = pd.DataFrame.from_dict(submission_dict, orient='index')
    sub.index.names = ['patientId']
    sub.columns = ['PredictionString']
    sub.to_csv(result_csv_path)
