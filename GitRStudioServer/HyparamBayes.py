#Hyperparameter search using keras-tuner
import keras_tuner
from tensorflow import keras


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
                                           validation_split = 0.3,
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
                                           validation_split = 0.3,
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
  
      # Rebuild top and use our data as top
      x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
      x = keras.layers.BatchNormalization()(x)
      
      ### Dropout tries to prevent overfitting
      top_dropout_rate = 0.2
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

tuner = keras_tuner.BayesianOptimization(
                       hypermodel=build_model,
                       objective = "val_accuracy", #optimize val acc
                       max_trials = 15, #for each candidate model
                       overwrite = True,  #overwrite previous results
                       directory = 'hyperband_search_dir', #Saving dir
                       project_name = 'FrozenLayersClassifier')
                       
                       
tuner.search(x=train_data, 
             epochs= 2,  # Max num of candidates to try
             validation_data=val_data)
            
models = tuner.get_best_models(num_models=2)
best_model = models[0]
best_model.build(input_shape=(600, 600, 3))
best_model.summary()
#### No  I need to know how to iterate over different models
