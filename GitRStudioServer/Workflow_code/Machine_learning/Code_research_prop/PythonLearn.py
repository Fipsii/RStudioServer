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






img_width = 256
img_height = 256
target_size = (img_width, img_height)
channels = 1

output_n = 2

path = "/Users/vg/Desktop/Daphnia"
train_image_files_path = path + "/Training"
valid_image_files_path = path + "/Test"

train_data_gen = ImageDataGenerator(rescale = 1./255)
valid_data_gen = ImageDataGenerator(rescale = 1./255)

train_image_array_gen = train_data_gen.flow_from_directory(train_image_files_path, shuffle = False,
                                                    target_size = target_size,
                                                    class_mode = 'categorical', 
                                                    classes = ("Magna", "Longicephala"))

valid_image_array_gen = valid_data_gen.flow_from_directory(valid_image_files_path, shuffle = False,
                                                    target_size = target_size,
                                                    class_mode = 'categorical', 
                                                    classes = ("Magna", "Longicephala"))

train_samples = train_image_array_gen.n
valid_samples = valid_image_array_gen.n

##### Model 



#### batch and epochs

batch_size = 32
epochs = 10

### initalize

model = Sequential()


# input layer
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', input_shape = (img_width, img_height, channels), activation = 'relu'))
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


RoundedY = np.argmax(ynew, axis = 1)

print("Confusion matrix:\n", confusion_matrix(valid_image_array_gen.classes,RoundedY))
target_names= list(valid_image_array_gen.class_indices.keys())
print("Confusion matrix:\n",classification_report(valid_image_array_gen.classes, RoundedY, target_names=target_names))



####### Heatmap approach




import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

##### Einzelbild das getestet wird
img = cv2.imread("/Users/vg/Desktop/D_longicephala/Test/magna/C3_05.tif")
img = cv2.resize(img, (256,256))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
X = np.expand_dims(img, axis=0).astype(np.float32)
X = preprocess_input(X)

### Collect needed output layers and import keras model (should later be our own! NO Flatten is important)


conv_output = model.get_layer("conv2d_5").output
pred_output = model.get_layer("dense_5").output
model.summary()
model = Model(model.input, outputs=[conv_output, pred_output])
conv, pred = model.predict(X)
decode_predictions(pred)
2048/16
######## Make gallery that shows each neurons interest area

scale = 256 / 7
plt.figure(figsize=(16, 16))
for i in range(36):
    plt.subplot(6, 6, i + 1)
    plt.imshow(img)
    plt.imshow(zoom(conv[0, :,:,i], zoom=(scale, scale)), cmap='jet', alpha=0.3)
    plt.show()
    
##### Make one image which shows the generak heatmap
    
target = np.argmax(pred, axis=1).squeeze()
w, b = model.get_layer("predictions").weights
weights = w[:, target].numpy()
heatmap = conv.squeeze() @ weights

scale = 224 / 7
plt.figure(figsize=(12, 12))
plt.imshow(img)
plt.show(zoom(heatmap, zoom=(scale, scale)), cmap='jet', alpha=0.5)




