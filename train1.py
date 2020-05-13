import json
import pickle

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from os import listdir

train_data_dir = r'data/train'
validation_data_dir = r'data/valid'
test_data_dir = r'data/test'
species_list = listdir(train_data_dir)
img_width, img_height = 224, 224

def get_empty_model():
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(180))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0001, decay=1e-6), metrics=['accuracy'])

    return model


def train_model(model, with_augmentation=True, save_history=True):
    #nb_train_samples = 24497
    #nb_validation_samples = 900
    epochs = 20  # 50
    batch_size = 128

    if with_augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,  #
            #zoom_range=0.2,  #
            rotation_range=15,  #
            horizontal_flip=True  #
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255
        )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        classes=species_list,
        #shuffle=True,
        class_mode='categorical')  # class_mode: One of "categorical", "binary", "sparse", "input", or None

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        classes=species_list,
        class_mode='categorical')

    history = model.fit_generator(
        train_generator,
        #steps_per_epoch=nb_train_samples // batch_size,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        #validation_steps=nb_validation_samples // batch_size
        validation_steps=validation_generator.n // batch_size
        # ,verbose = 2
        # ,callbacks = [ModelCheckpoint(filepath="checkpoints.h5", verbose=1, save_best_only=True)]  # save best model after every epoch
    )

    model.save_weights('first_try.h5')

    if save_history:
        json.dump(history.history,
            open("history.json.txt", 'w')  # history_dict = json.load(open("history.json.txt", 'r'))
        )


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
    train_model(get_empty_model(), save_history=True, with_augmentation=True)
    #test_model_on_test_data(load_model('model2_noaug_first_try.h5'))
    #test_model_on_test_data(load_model('first_try.h5'))
