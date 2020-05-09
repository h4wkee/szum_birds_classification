import json

from keras.optimizers import rmsprop
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from os import listdir

train_data_dir = r'..\100-bird-species\180\train'
validation_data_dir = '../100-bird-species/180/valid'
species_list = listdir(train_data_dir)
img_width, img_height = 150, 150 # 224, 224

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
model1.add(Dense(256)) #64
model1.add(Activation('relu'))
model1.add(Dropout(0.5))
model1.add(Dense(180))
model1.add(Activation('sigmoid'))

model1.compile(loss='categorical_crossentropy',  # 'binary_crossentropy'
              optimizer='rmsprop',
               metrics=['accuracy'])

#another model, from https://www.shirin-glander.de/2018/06/keras_fruits/
#but model1 is probably better
model2 = Sequential()
model2.add(Conv2D(32, (3, 3), input_shape=input_shape))
model2.add(Activation('relu'))

model2.add(Conv2D(16, (3, 3)))
model2.add(Activation('relu')) #leaky_relu(0.5)
model2.add(BatchNormalization())

model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Flatten())
model2.add(Dense(256))
model2.add(Activation('relu'))
model2.add(Dropout(0.5))

model2.add(Dense(180))
model2.add(Activation('softmax'))

model2.compile(loss='categorical_crossentropy', optimizer=rmsprop(lr = 0.0001, decay = 1e-6), metrics=['accuracy'])


##################

model = model1

nb_train_samples = 24497
nb_validation_samples = 900
epochs = 10 # 50
batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,        #optional augmentation
    zoom_range=0.2,         #optional augmentation
    horizontal_flip=True    #optional augmentation
    )

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    classes = species_list,
    class_mode='categorical')  # class_mode: One of "categorical", "binary", "sparse", "input", or None

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    classes = species_list,
    class_mode='categorical')


history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')

# plt.plot(history)
# or save to file to plot later
with open('history.txt', 'w') as filehandle:
    json.dump(history, filehandle)


# another kind of fit, probably the same
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch = train_generator.n // batch_size,
#     epochs = epochs,
#     validation_data = validation_generator,
#     validation_steps = validation_generator.n // batch_size,
#     verbose = 2,
#     #callbacks = [
#     # # save best model after every epoch
#     # callback_model_checkpoint("checkpoints.h5", save_best_only=TRUE)]
# )