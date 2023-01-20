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

import os as os
from PIL import Image


##### Convert all tifs to png. This step should be in a different document#####
path = "/Users/vg/Desktop/Daphnia_Raw/pulex"
current_path = path

for root, dirs, files in os.walk(current_path, topdown=False):
  print(dirs, files)
  for name in files:
    print(os.path.join(root, name))
    if os.path.splitext(os.path.join(root, name))[1].lower() == ".tif":
      if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".png"):
        print ("A jpeg file already exists for %s")
      else:
        outfile = os.path.splitext(os.path.join(root, name))[0] + ".png"
        try:
          im = Image.open(os.path.join(root, name))
          print ("Generating png for %s" )
          im.thumbnail(im.size)
          im.save(outfile, "PNG", quality=100)
        except Exception:
          print("Except")



#### Initilaize data 224x224 is the biggest possible

img_width = 224
img_height = 224
target_size = (img_width, img_height)
output_n = 4 ## Number of classes
path = "/Users/vg/Desktop/Daphnia_Raw"

#### Create train and vaildation set val split amount val data, seed has to be the same

train_data = keras.utils.image_dataset_from_directory(path,  
                                         image_size = target_size,
                                         labels = "inferred",
                                         color_mode = "rgb",
                                         label_mode = 'categorical', 
                                         subset = "training",
                                         class_names = ("magna", "longicephala", "pulex","cucullata"),
                                         validation_split = 0.1,
                                         batch_size = 32,
                                         seed = 634
                                         )

val_data = keras.utils.image_dataset_from_directory(path, 
                                         image_size = target_size,
                                         labels = "inferred",
                                         color_mode = "rgb",
                                         label_mode = 'categorical', 
                                         subset = "validation",
                                         class_names = ("magna", "longicephala", "pulex","cucullata"),
                                         validation_split = 0.1,
                                         batch_size = 32,
                                         seed = 634
                                         )
                                         
### Standardize### Augment data and create test_data
val_batches = tf.data.experimental.cardinality(val_data).numpy()
print(val_batches)
print(tf.data.experimental.cardinality(train_data).numpy())

test_data = val_data.take(val_batches // 2)
val_data = val_data.skip(val_batches // 2)

normalization_layer = keras.layers.Rescaling(1./255)


## Optimize data reading by prebuffering some data and not all

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(buffer_size=AUTOTUNE)
val_data = val_data.prefetch(buffer_size=AUTOTUNE)
#test_data = test_dataset.prefetch(buffer_size=AUTOTUNE)

####  Flip data randomly

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

##### Model transfer learned ### Needs jpg bmp png gif or jpeg

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

IMG_SHAPE = target_size + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
                                               
image_batch, label_batch = next(iter(train_data))
np.shape(label_batch)
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.summary()


####### Freeze ###### 

base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)

prediction_layer = tf.keras.layers.Dense(4)
prediction_batch = prediction_layer(feature_batch_average)

inputs = tf.keras.Input(shape=(224, 224, 3))
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

loss0, accuracy0 = model.evaluate(val_data)

history = model.fit(train_data,
                    epochs=initial_epochs,
                    validation_data=val_data)
                                               
###### Learning curve ## Frozen weights
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()



############ Fine tune and update weights unfreeze and all


base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False
  
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
              metrics=['accuracy'])

model.summary()

fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_data,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=val_data)
                         
### Plot again with new learned weights

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.1, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


###### Test the data

loss, accuracy = model.evaluate(test_data)
print('Test accuracy :', accuracy)
predictions = model.predict(test_data)

np.shape(predictions)
from sklearn.metrics import classification_report, confusion_matrix

#### Save model ##

model.save("/Users/vg/Desktop/Daphnia_Model")
####### Confusion be confused ##### Does not match up with accuracy! Both methods of label extraction have different values Porbbabaly due to some shiúffeling in the data

### Look into test_data

Looki = test_data.unbatch()
Looki

images, labels = tuple(zip(*test_data.unbatch()))
np.shape(labels)
np.array(labels)
Real_labels = np.argmax(labels, axis = 1)

true_categories = tf.concat([y for x, y in test_data], axis=0)
true_categories = np.argmax(true_categories, axis = 1)
Predicted_labels = np.argmax(predictions, axis = 1)

print("Confusion matrix:\n", confusion_matrix(true_categories,Real_labels))
target_names = ("magna", "longicephala", "pulex","cucullata")
print("Classification report:\n",classification_report(true_categories, Predicted_labels, target_names=target_names))


# =============================================
# Grad-CAM code
# =============================================
import numpy as np
import matplotlib.pyplot as plt
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.activation_maximization.callbacks import Progress
from tf_keras_vis.activation_maximization.input_modifiers import Jitter, Rotate2D
from tf_keras_vis.activation_maximization.regularizers import TotalVariation2D, Norm
from tf_keras_vis.utils.model_modifiers import ExtractIntermediateLayer, ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tensorflow.keras.preprocessing.image import load_img
import cv2 as cv2


image_title = "Daphnia_Magna"
Magna = load_img("/Users/vg/Desktop/Daphnia/magna/DeTaVw_softver thingy_82.png", color_mode = "rgb")
Pulex = load_img("/Users/vg/Desktop/Daphnia/pulex/Po12_189.png", color_mode = "rgb")
Cucullata = load_img("/Users/vg/Desktop/Daphnia/cucullata/D.cucullata_2017-07-18_I1_W (1).png", color_mode = "rgb")
Longhicephala = load_img("/Users/vg/Desktop/Daphnia/longicephala/C-13.png", color_mode = "rgb")

List_of_Images = [Magna,Pulex,Cucullata, Longhicephala]
Processed_Images = []

for x in List_of_Images:
  img_width = 224
  img_height = 224
  target_size = (img_width, img_height)
  X = cv2.resize(np.array(x, dtype = "float32"), target_size)
  #X = np.reshape(X, (224,224,-1))
  Processed_Images.append(X)
  

from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

replace2linear = ReplaceToLinear()

score = CategoricalScore(0)


from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency
# from tf_keras_vis.utils import normalize

# Create Saliency object.
saliency = Saliency(model,
                    model_modifier=replace2linear,
                    clone=True)

# Generate saliency map
Saliency_maps = []

for x in Processed_Images:
  saliency_map = saliency(score, x, smooth_samples= 20, smooth_noise= 0.05) # noise spread level.
  Saliency_maps.append(saliency_map)
# Render

type(saliency_map)
fig, axs = plt.subplots(2, 2)


plt.figure(figsize = (2,2))

for x in range(len(Saliency_maps)):
  plt.imshow(Saliency_maps[x][0])
  plt.imshow(Processed_Images[x].astype(np.uint8), cmap = "gray", alpha = 0.45)
  plt.show()








