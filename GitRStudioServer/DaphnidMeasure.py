### This file contains functions to evaluate the distances in Pixel and
### visualize the measured distances for control and visualization

def DaphnidMeasurements(DataframewithLabelledData):
  import numpy as np
  import pandas as pd
  #### Changing copy warning: Right now: ignored
  ### Coco box defintion 'bbox': [x_min, y_min, width, height]
  ### -> X_max, Y_max = X_min + width, Y_max + height
  #### Now we have a dataframe containing two points per box and a row per animal
  #### We calculate now the midpoints of all of them: (x₁ + x₂)/2, (y₁ + y₂)/2
  ## Eyes
  data = pd.read_csv(DataframewithLabelledData)
  
  data["Xmax_Eye"] = data["Xmin_Eye"] + data["bboxWidth_Eye"]
  data["Ymax_Eye"] = data["Ymin_Eye"] + data["bboxHeight_Eye"]
  data["Center_X_Eye"] = (data["Xmax_Eye"] + data["Xmin_Eye"])/2
  data["Center_Y_Eye"] = (data["Ymax_Eye"] + data["Ymin_Eye"])/2
  ## Spina base
  
  
  data["Xmax_Spina_base"]  = data["Xmin_Spina base"] + data["bboxWidth_Spina base"]
  data["Ymax_Spina_base"] = data["Ymin_Spina base"] + data["bboxHeight_Spina base"]
  data["Center_X_Sb"] = (data["Xmax_Spina_base"] + data["Xmin_Spina base"])/2
  data["Center_Y_Sb"] = (data["Ymax_Spina_base"] + data["Ymin_Spina base"])/2
  ## Spina tip
  
  
  data["Xmax_Spina_tip"] = data["Xmin_Spina tip"] + data["bboxWidth_Spina tip"]
  data["Ymax_Spina_tip"] = data["Ymin_Spina tip"] + data["bboxHeight_Spina tip"]
  data["Center_X_St"] = (data["Xmax_Spina_tip"] + data["Xmin_Spina tip"])/2
  data["Center_Y_St"] = (data["Ymax_Spina_tip"] + data["Ymin_Spina tip"])/2
  
  ### Merge the datasets with nan values if measurements do not exist for an individual
  #### Calculate body length
  ### Distance sqrt((x1-x2)^2 +(y1-y2)^2)
  
  data["Bodylength[px]"] = np.sqrt((data["Center_X_Sb"]-data["Center_X_Eye"])**2 + (data["Center_Y_Sb"]-data["Center_Y_Eye"])**2)
  data["Spinalength[px]"] = np.sqrt((data["Center_X_Sb"]-data["Center_X_St"])**2 + (data["Center_Y_Sb"]-data["Center_Y_St"])**2)
  
  data.to_csv(DataframewithLabelledData, index = False)
  ### Plot an tested Image: Idea make a folder with all Images for safety? Necessary? Images were checked already while relabeling!
  
  return("px Werte eingetragen")
### To do  DPI -> px to mm as well as producing csv output and Clean_up output csv Name


### A funtion that plots points for a given Image
### Imagepath is the path to Image that should be edited
### Dataframe ist the Dataframe resulting from Daphnid Measurements
### Image should NOT be renamed as we use the name to find the corresponding Image

def ShowMeasureImage(Imagepath, Dataframe):
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  plt.clf()
  im = plt.imread(Imagepath)
  plt.imshow(im)
  ### split the path:

  filename = Imagepath.split("/")[-1]
 
  Image_index = data.index[data['image_id'] == filename]
  Image_index = Image_index[0] ## Make Int64IndexSeries of one number into a number
  ### Plot the lines
  plt.plot([data["Center_X_Sb"][Image_index],data["Center_X_Eye"][Image_index]],[data["Center_Y_Sb"][Image_index],data["Center_Y_Eye"][Image_index], color = "red", linewidth=0.5)
  plt.plot([data["Center_X_Sb"][Image_index],data["Center_X_St"][Image_index]],[data["Center_Y_Sb"][Image_index],data["Center_Y_St"][Image_index]], color = "red", linewidth=0.5) 
  ### Plot the points
  plt.plot(data["Center_X_Eye"][Image_index],data["Center_Y_Eye"][Image_index],marker="o", markersize=2, markeredgecolor="black", markerfacecolor="white")
  plt.plot(data["Center_X_Sb"][Image_index],data["Center_Y_Sb"][Image_index],marker="o", markersize=2, markeredgecolor="black", markerfacecolor="white") 
  plt.plot(data["Center_X_St"][Image_index],data["Center_Y_St"][Image_index],marker="o", markersize=2, markeredgecolor="black", markerfacecolor="white") 
  plt.show()
  
  return("Image of" + filename + "printed")


DaphnidMeasurements("/home/philipp//GitRStudioServer/ImageData/SimonaRelabelled/annotations/instances_default.csv")
