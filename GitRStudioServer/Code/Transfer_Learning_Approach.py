import keras as keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import (Conv2D,
                          Dense,
                          LeakyReLU,
                          BatchNormalization, 
                          MaxPooling2D, 
                          Dropout,
                          Flatten)
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard
import PIL.Image
from datetime import datetime as dt
import tensorflow as tf

import os
from PIL import Image

path = "/Users/vg/Desktop/Daphnia"
current_path = train_image_files_path 


for root, dirs, files in os.walk(current_path, topdown=False):
  print(dirs, files)
  for name in files:
    print(os.path.join(root, name))


im = Image.open("/Users/vg/Desktop/Daphnia/Magna_Überschuss/Control1.tif")
print ("Converting png for %s", im)
im.convert("L")
im.save("/Users/vg/Desktop/Daphnia/Magna_Überschuss/Control1.png", format = "PNG", quality=100)
im.show()



outputfile
img_width = 160
img_height = 160
target_size = (img_width, img_height)

output_n = 4

path = "/Users/vg/Desktop/Daphnia"
train_image_files_path = path + "/Training"
valid_image_files_path = path + "/Test"

train_data_gen = ImageDataGenerator(rescale = 1./255)
valid_data_gen = ImageDataGenerator(rescale = 1./255)

train_image_array_gen = train_data_gen.flow_from_directory(train_image_files_path, shuffle = False,
                                                           target_size = target_size,
                                                           color_mode = "grayscale",
                                                           class_mode = 'categorical', 
                                                           classes = ("Magna", "Longicephala", "pulex","cucullata"),
                                                           )

valid_image_array_gen = valid_data_gen.flow_from_directory(valid_image_files_path, shuffle = False,
                                                           target_size = target_size,
                                                           color_mode = "grayscale",
                                                           class_mode = 'categorical', 
                                                           classes = ("Magna", "Longicephala", "pulex","cucullata"),
                                                           )

#tf.keras.utils.image_dataset_from_directory Not deprecated but can't read tifs. Convert images?

type(train_image_array_gen._variant_tensor)
train_samples = train_image_array_gen.n
valid_samples = valid_image_array_gen.n
class_names = list(train_image_array_gen.class_indices.keys())

##### Model transfer learned ### Needs jpg bmp png gif or jpeg
s
val_batches = tf.data.experimental.cardinality(valid_image_array_gen)
test_dataset = valid_image_array_gen.take(val_batches // 5)
validation_dataset = valid_image_array_gen.skip(val_batches // 5)

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

IMG_SHAPE = target_size + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
                                               
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)

inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
initial_epochs = 10

loss0, accuracy0 = model.evaluate(validation_dataset)

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)
                                               
