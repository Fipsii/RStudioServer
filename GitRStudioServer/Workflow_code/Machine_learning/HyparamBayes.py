#Hyperparameter search using keras-tuner ## We try to find the best hyperparameters for our unfrozen Model
# We search for layers, lr and Dropout
# Cross validation with keras_tuner.tuners.SklearnTuner
import keras_tuner
from tensorflow import keras
from keras.layers import (Conv2D,
                          Dense,
                          LeakyReLU,
                          BatchNormalization, 
                          MaxPooling2D, 
                          Dropout,
                          Flatten)
from sklearn import ensemble
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection

### Build Model how to combine with unfreeze?
img_width = 600
img_height = 600
channels = 3
target_size = (img_width, img_height)
output_n = 5 ## Number of classes
path = "/home/philipp/GitRStudioServer/Data_DaphniasPNG_entpackt"

train_data = keras.utils.image_dataset_from_directory(path,  
                                           image_size = target_size,
                                           labels = "inferred",
                                           color_mode = "rgb",
                                           label_mode = 'categorical', 
                                           subset = "training",
                                           class_names = ("magna", "longicephala", "pulex","cucullata", "longispina"),
                                           validation_split = 0.01,
                                           batch_size = 32,
                                           seed = 634
                                           )
  
val_data = keras.utils.image_dataset_from_directory(path, 
                                           image_size = target_size,
                                           labels = "inferred",
                                           color_mode = "rgb",
                                           label_mode = 'categorical', 
                                           subset = "validation",
                                           class_names = ("magna", "longicephala", "pulex","cucullata", "longispina"),
                                           validation_split = 0.01,
                                           batch_size = 32,
                                           seed = 634
                                           )
def build_model(hp):
      #### Building a Model for EfficientNET
      #### Inputs are the expected shapes: Here 600,600,3 and get resized
      METRICS = ['accuracy',
      keras.metrics.AUC(name='auc'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall')
      ]
      inputs = keras.layers.Input(shape=(600, 600, 3))
      #### Here Augment our Images using our self defined augemtation values FLip, transform, rotate
      x = inputs
      #x = img_augmentation(inputs)
      ### initalize EfficientNETB7 for transfer learning, exclude Top, 
      ### Augmented Data as input_tensor, shape as in inputs, use imagenet weights
      model = keras.applications.EfficientNetB7(include_top=False, input_tensor=x, input_shape = (600,600,3), weights="imagenet")
  
      # Freeze the pretrained weights
      model.trainable = False
      
      for layer in model.layers[-20:]:
          if not isinstance(layer, keras.layers.BatchNormalization):
              layer.trainable = True
        

  
      # Rebuild top and use our data as top
      x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
      x = keras.layers.BatchNormalization()(x)
      
      ### Dropout tries to prevent overfitting
      top_dropout_rate = hp.Float("dropout", min_value = 0, max_value = 0.5, sampling = "linear")
      x = keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x) 
      outputs = keras.layers.Dense(output_n, activation="softmax", name="pred")(x)
  
      # Compile
      model = keras.Model(inputs, outputs, name="EfficientNetB7")
      
      ### now do hyperparam search
      
      lr = hp.Float("lr", min_value = 0.0001, max_value = 0.01, sampling = "log")
      optimizer = keras.optimizers.Adam(lr)
      model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=METRICS)       
      return model


# defining the number of neurons in a fully connected layer

tuner = keras_tuner.tuners.SklearnTuner(
                       oracle = keras_tuner.oracles.BayesianOptimizationOracle(
                                objective=keras_tuner.Objective('score', 'max'),
                                max_trials=10),
                       scoring = metrics.make_scorer(metrics.accuracy_score),
                       cv = model_selection.StratifiedKFold(5),
                       directory = 'hyperband_search_dir',
                       project_name = 'cross_validated',
                       hypermodel= build_model, #optimize val acc
                       overwrite = True  #overwrite previous results
                       )

### For cross validation we have to read in our data differently
from os import listdir
from os.path import isfile, join
import os
import re
import matplotlib.pyplot as plt

mypath = '/home/philipp/GitRStudioServer/Data_DaphniasPNG_entpackt' # edit with the path to your data
files = os.listdir(mypath)

## Drop DS.Store
files = files[:-1]
image_arrays = []
labels = []

for name in files:
    temp_path = mypath + "/"+ name
    list_of_files = os.listdir(temp_path)

    for image in list_of_files:
      try:
        img = plt.imread(temp_path + "/" + image)
        image_arrays.append(img) 
        labels.append(name)
      except:
        print(image + " is not an image file")

import numpy as np
labels = np.array(labels)
image_arrays = np.array(image_arrays)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(image_arrays, labels, test_size=0.2)
X, y = datasets.load_iris(return_X_y=True)

tuner.search(X_train, Y_train)
