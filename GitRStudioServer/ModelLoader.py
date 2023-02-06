import keras as keras
import tensorflow as tf
import PIL as PIL

img = PIL.Image.open("Code/model.png")

### Load_Model
##"/home/philipp/GitRStudioServer/SavedModels/MobileV2ModelefficientNetB7"
path = "/home/philipp/GitRStudioServer/SavedModels/Daphnia_ModelVGG16"
model = keras.models.load_model(path)
model
img_width = 224
img_height = 224
target_size = (img_width, img_height)
#### Get the data test data 
#### for this we load the val data wit the same seed as in the original model. We can with the same function load different images
path = "/home/philipp/GitRStudioServer/Data_DaphniasPNG_entpackt"

val_data = keras.utils.image_dataset_from_directory(path, 
                                           image_size = target_size,
                                           labels = "inferred",
                                           color_mode = "rgb",
                                           label_mode = 'categorical', 
                                           subset = "validation",
                                           class_names = ("magna", "longicephala", "pulex","cucullata", "longispina"),
                                           validation_split = 0.1,
                                           batch_size = 32,
                                           seed = 634
                                           )
  
##### val_batches is the amount of batches used for validation
##### We then split the data. Every second batch is for testing every other for validation
val_batches = tf.data.experimental.cardinality(val_data).numpy()
test_data = val_data.take(val_batches // 2)
val_data = val_data.skip(val_batches // 2)    

### Now we have test data in the right format for model evaluation

### Plot again with new learned weights
###### Test the data
LossAccList = []
loss, accuracy = model.evaluate(test_data)
print('Test accuracy :', accuracy)
predictions = model.predict(test_data)
print('Predictions :', np.argmax(predictions))

temp =[loss, accuracy, predictions]
LossAccList = LossAccList.append(temp)


##### Here we want to additionally visualize what our network sees. For this we use tf_keras_vis to create a heatmap, which
##### uses our weights and plots them into images we provide

### Read in example images for evaluation and make a list
Magna = load_img("Data_DaphniasPNG_entpackt/magna/Project_D.magna_Aig_1.png", color_mode = "rgb")
Pulex = load_img("Data_DaphniasPNG_entpackt/pulex/D_pulex_A6_2_Verg1.6.png", color_mode = "rgb")
Cucullata = load_img("Data_DaphniasPNG_entpackt/cucullata/D_cucullata_TSBR_19_Verg4.0_Licht_oben_unten_without_eggs.png", color_mode = "rgb")
Longhicephala = load_img("Data_DaphniasPNG_entpackt/longicephala/Longicephala_Handy (3).jpg", color_mode = "rgb")
Longispina = load_img("Data_DaphniasPNG_entpackt/longispina/Longispina_Handy (2).jpg", color_mode = "rgb")

List_of_Images = [Magna,Pulex,Cucullata, Longhicephala, Longispina]
Processed_Images = []

## Make a list of these images as an array
for x in List_of_Images:
  img_width = 600
  img_height = 600
  target_size = (img_width, img_height)
  X = cv2.resize(np.array(x, dtype = "float32"), target_size)
  Processed_Images.append(X)

model.predict(Processed_Images[0])  

#### Create the saliency heatmap this is based on the last weights

from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency

replace2linear = ReplaceToLinear()
score = CategoricalScore(0)

# Create Saliency object.
saliency = Saliency(model,
                    model_modifier=replace2linear,
                    clone=True)

# Generate saliency map
Saliency_maps = []

for x in Processed_Images:
  saliency_map = saliency(score, x, smooth_samples= 20, smooth_noise= 0.05) # noise spread level.
  Saliency_maps.append(saliency_map)

for x in range(len(Saliency_maps)):
  
  plt.clf()
  figs, axs = plt.subplots(1,2)
  figs.suptitle("Daphnia Image: " + "Ep: 5;20")
  axs[0].axis("off")
  axs[0].imshow(Saliency_maps[x][0])
  axs[1].axis("off")
  axs[1].imshow(Processed_Images[x].astype(np.uint8), cmap = "gray", alpha = 1)
  
  #plt.imshow(Processed_Images[x].astype(np.uint8), cmap = "gray", alpha = 0.45)
  plt.show()


keras.utils.vis_utils.plot_model(model, to_file = "EfficientNETB7.png")
