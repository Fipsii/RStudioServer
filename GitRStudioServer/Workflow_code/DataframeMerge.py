### This Code is planned to merge the two existing Dataframes (Measurments and Scale)
### and calculate mm values ### Important to note: We should set paths from configs


### Test run with simonas data ### Make new Datamerge.py later


def FuseDataframes(dfPixel,dfScales, savelocation):
  import pandas as pd

  dfPixel = dfPixel.rename(columns={'image_id': 'Name'})
  DataFinished = pd.DataFrame()
  DataFinished = pd.merge(dfPixel,dfScales, on = "Name", how = "inner")
  DataFinished["Spinalength[mm]"] = DataFinished["Spinalength[px]"] * DataFinished["distance_per_pixel"]
  DataFinished["Bodylength[mm]"] = DataFinished["Bodylength[px]"] * DataFinished["distance_per_pixel"]
  DataFinished.to_csv(save, index = False)
  return DataFinished
### This Works now we have to sort out the doubles and false positives (mostly 0s that are easy to discard) 
### We also have to discuss the few false line values in the second funtion

def ShowMeasureImage(Imagepath, data, visualize): ### Input the dataframe and 1 Image
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import os as os
  import cv2 as cv2
  if visualize == False:
    print("no visualization mode")
  elif visualize == True:
    #### Mode True visualize a single Image or a list of Images and save them
    save_folder = Imagepath[x].split("/")[:-1] + "Measurement_Visualization"
    os.mkdir(save_folder)
    plt.clf()
    plt.figure(figsize = (2560,1600))
    if type(Imagepath) != list: ## Make a list if only one Image is submitted
      Imagepath = [Imagepath]
    
    for x in Imagepath:
      im = plt.imread(Imagepath[x])
      plt.imshow(im)
    
      
    ### split the path:
  
      filename = Imagepath[x].split("/")[-1]
     
      Image_index = data.index[data['image_id'] == filename]
      Image_index = Image_index[0] ## Make Int64IndexSeries of one number into a number
      ### Body with still needs to implemented
      ### Plot the lines
      plt.plot([data["Center_X_Sb"][Image_index],data["Center_X_Eye"][Image_index]],[data["Center_Y_Sb"][Image_index],data["Center_Y_Eye"][Image_index]], color = "red", linewidth=0.5)
      plt.plot([data["Center_X_Sb"][Image_index],data["Center_X_St"][Image_index]],[data["Center_Y_Sb"][Image_index],data["Center_Y_St"][Image_index]], color = "red", linewidth=0.5) 
      ### Plot the points
      plt.plot(data["Center_X_Eye"][Image_index],data["Center_Y_Eye"][Image_index],marker="o", markersize=2, markeredgecolor="black", markerfacecolor="white")
      plt.plot(data["Center_X_Sb"][Image_index],data["Center_Y_Sb"][Image_index],marker="o", markersize=2, markeredgecolor="black", markerfacecolor="white") 
      plt.plot(data["Center_X_St"][Image_index],data["Center_Y_St"][Image_index],marker="o", markersize=2, markeredgecolor="black", markerfacecolor="white") 
      plt.axis('off')
      plt.show()
      plt.savefig(save_folder +"/visualization_of_" + filename, dpi=1000,bbox_inches= "tight",pad_inches=0)
      print("Visualization of" + filename + " printed")
  return()

### Which visualization options? True = Image or Image list and False = No
from Yaml_load_test import ConfigImport
import pandas as pd
settings = ConfigImport("/home/philipp/GitRStudioServer/Workflow_code/settings.yml")
dfPixel = pd.read_csv(settings["Annotation_path"][:-5]+".csv")
dfScales = pd.read_csv(settings["Working_folder_path"] + "Scale.csv")
path = path = settings["Images_path"]
save = settings["Working_folder_path"] + "Datafinished.csv"

#Paths_of_Images, Name_of_Images = Images_list(path)
DataFrame = FuseDataframes(dfPixel, dfScales, save)
#ShowMeasureImage(DataFrame,True)


