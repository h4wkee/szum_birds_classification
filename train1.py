import json
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

from keras.optimizers import rmsprop
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from os import listdir

train_data_dir = r'data/train'
validation_data_dir = r'data/valid'
test_data_dir = r'data/test'
species_list = listdir(train_data_dir)
img_width, img_height = 150, 150  # 224, 224

subject = 'birds'
split = 8
epochs = 16
lr_rate = .006
image_size = 224
model_size = 's'
dropout = .5
rand_seed = 12357
dwell = False
kaggle = True


def get_empty_model():
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model1 = Sequential()
    model1.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))

    model1.add(Conv2D(32, (3, 3)))
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))

    model1.add(Conv2D(64, (3, 3)))
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))

    model1.add(Flatten())
    model1.add(Dense(256))  # 64
    model1.add(Activation('relu'))
    model1.add(Dropout(0.5))
    model1.add(Dense(180))
    model1.add(Activation('sigmoid'))

    model1.compile(loss='categorical_crossentropy',  # 'binary_crossentropy'
                   optimizer='rmsprop',
                   metrics=['accuracy'])

    # another model, from https://www.shirin-glander.de/2018/06/keras_fruits/
    model2 = Sequential()
    model2.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model2.add(Activation('relu'))

    model2.add(Conv2D(16, (3, 3)))
    model2.add(Activation('relu'))  # leaky_relu(0.5)
    model2.add(BatchNormalization())

    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.25))

    model2.add(Flatten())
    model2.add(Dense(256))
    model2.add(Activation('relu'))
    model2.add(Dropout(0.5))

    model2.add(Dense(180))
    model2.add(Activation('softmax'))

    model2.compile(loss='categorical_crossentropy', optimizer=rmsprop(lr=0.0001, decay=1e-6), metrics=['accuracy'])

    return model2


def make_model(classes, lr_rate, image_size, model_size, dropout, rand_seed):
    size = len(classes)
    mobile = tf.keras.applications.mobilenet.MobileNet(include_top=False,
                                                       input_shape=(image_size, image_size, 3),
                                                       pooling='max', weights='imagenet',
                                                       alpha=1, depth_multiplier=1, dropout=.4)
    x = mobile.layers[-1].output
    if model_size == 'S':
        pass
    # x=Dense(128, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),
    # bias_regularizer=regularizers.l1(0.006) ,activation='relu')(x)
    # x=Dropout(rate=dropout, seed=rand_seed)(x)
    elif model_size == 'M':
        x = Dense(256, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
                  bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
        x = Dropout(rate=dropout, seed=rand_seed)(x)
        x = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
        x = Dense(16, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
                  bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
        x = Dropout(rate=dropout, seed=rand_seed)(x)
    else:
        x = Dense(1024, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
                  bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
        x = Dropout(rate=dropout, seed=rand_seed)(x)
        x = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
        x = Dense(128, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
                  bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
        x = Dropout(rate=dropout, seed=rand_seed)(x)
        x = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
        x = Dense(16, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
                  bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
        x = Dropout(rate=dropout, seed=rand_seed)(x)
    x = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    predictions = Dense(len(classes), activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=predictions)
    for layer in model.layers:
        layer.trainable = True
    model.compile(Adamax(lr=lr_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, with_augmentation=True, save_history=False):
    nb_train_samples = 24497
    nb_validation_samples = 900
    epochs = 10  # 50
    batch_size = 16

    if with_augmentation:
        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,  # optional augmentation
            zoom_range=0.2,  # optional augmentation
            horizontal_flip=True  # optional augmentation
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        classes=species_list,
        class_mode='categorical')  # class_mode: One of "categorical", "binary", "sparse", "input", or None

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        classes=species_list,
        class_mode='categorical')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size
        # ,verbose = 2
        # ,callbacks = [ModelCheckpoint(filepath="checkpoints.h5", verbose=1, save_best_only=True)]  # save best model after every epoch
    )

    model.save_weights('first_try.h5')

    if save_history:
        # don't know if it works, don't know which method is better
        with open("history.pickled.txt", "wb") as filehandle:  # with open("history.pickled.txt", "rb") as filehandle:
            pickle.dump(history.history, filehandle)  # history = pickle.load(filehandle)
        json.dump(history.history,
                  open("history.json.txt", 'w'))  # history_dict = json.load(open("history.json.txt", 'r'))


def load_model(filename='first_try.h5'):
    model = get_empty_model()
    model.load_weights(filename)
    return model


def test_model_on_test_data(model):
    batch_size = 16

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        classes=species_list,
        class_mode='categorical')

    score = model.evaluate_generator(test_generator, steps=test_generator.n // batch_size, verbose=2, callbacks=[])
    print(score)
    print(model.metrics_names)


if __name__ == '__main__':
    #train_model(get_empty_model())
    train_model(make_model(species_list, lr_rate, image_size, model_size, dropout, rand_seed))
    # test_model_on_test_data(load_model('model2_noaug_first_try.h5'))
