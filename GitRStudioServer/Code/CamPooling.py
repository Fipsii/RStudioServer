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






img_width = 600
img_height = 600
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

train_samples = train_image_array_gen.n
valid_samples = valid_image_array_gen.n

##### Model 



#### batch and epochs

batch_size = 32
epochs = 10

### initalize

model = Sequential()


# input layer
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', input_shape = (img_width, img_height, 1), activation = 'relu'))
# hiddel conv layer
model.add(Conv2D(filters = 16, kernel_size = (3,3), padding = 'same'))

model.add(LeakyReLU(.5))
model.add(BatchNormalization())
# using max pooling
model.add(MaxPooling2D(pool_size = (2,2)))
# randomly switch off 25% of the nodes per epoch step to avoid overfitting
model.add(Dropout(.25))
# flatten max filtered output into feature vector
model.add(Flatten())
# output features onto a dense layer
model.add(Dense(units = 100, activation = 'relu'))
# randomly switch off 25% of the nodes per epoch step to avoid overfitting
model.add(Dropout(.5))
# output layer with the number of units equal to the number of categories
model.add(Dense(units = output_n, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', 
              metrics = ['accuracy'], 
              optimizer = RMSprop(learning_rate = 1e-4, decay = 1e-6))

import keras.utils.vis_utils as kv
kv.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# fit
hist = model.fit(
  # training data
  
  x = train_image_array_gen,
  epochs = epochs,
  batch_size = batch_size,
  
  
)

############ Model evaluation ### training loss of model normally differs from evaluate loss (optimization and different behavior of layers in testing v training)

from sklearn.metrics import classification_report, confusion_matrix

Prediction = model.predict(valid_image_array_gen)
Prediction


RoundedY = np.argmax(Prediction, axis = 1)

print("Confusion matrix:\n", confusion_matrix(valid_image_array_gen.classes,RoundedY))
target_names= list(valid_image_array_gen.class_indices.keys())
print("Confusion matrix:\n",classification_report(valid_image_array_gen.classes, RoundedY, target_names=target_names))






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

model.summary()

image_title = "Daphnia_Magna"
img1 = load_img("/Users/vg/Desktop/Daphnia/Training/magna/C3_07.tif", color_mode = "grayscale")

img_width = 600
img_height = 600
target_size = (img_width, img_height)
X = cv2.resize(np.array(img1, dtype = "float32"), target_size)
X = np.reshape(X, (600,600,-1))

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
saliency_map = saliency(score, X, 
                        smooth_samples= 20, # The number of calculating gradients iterations.
                        smooth_noise= 0.05) # noise spread level.
# Render

type(saliency_map)

plt.imshow(saliency_map[0])
plt.imshow(X, cmap = "gray", alpha = 0.25)
plt.show()


##### SUperimposed
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize

gradcam = Gradcam(model,
                  model_modifier=replace2linear,
                  clone=True)

cam = gradcam(score,
              X,
              penultimate_layer=-1)

cam = normalize(cam)

heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)

plt.imshow(cam[0])
plt.imshow(heatmap, cmap = 'jet', alpha = 0.05)
plt.show()


