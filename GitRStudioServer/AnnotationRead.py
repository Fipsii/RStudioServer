##### Read in jsonfile and convert to pratical format
import json as json
import pandas as pd
### Read json into a dict

#filename = "/home/philipp/GitRStudioServer/ImageData/SimonaRelabelled/annotations/instances_default.json"
filename = "/home/philipp/GitRStudioServer/annotations3.json"
def JsonToMeasurement(filename):
  
  data = json.load(open(filename))
  ann = data["annotations"]
  Image_Ids = data["images"]
  Label_Ids = data["categories"]
  Imageframe = pd.DataFrame(Image_Ids)
  Labelframe = pd.DataFrame(Label_Ids)
  Annotationframe = pd.DataFrame(ann)

  ### Make ids into a readable format in the csv

  for x in range(1,len(Labelframe["name"])+1):
    Annotationframe["category_id"] = Annotationframe["category_id"].replace(x, Labelframe["name"][x-1])

  ### Same for Images ids

  for x in range(1,len(Imageframe["file_name"])+1):
    Annotationframe["image_id"] = Annotationframe["image_id"].replace(x, Imageframe["file_name"][x-1])

  #### Dewrangle the coordinates

  Annotationframe[['Xmin','Ymin','bboxWidth','bboxHeight']] = pd.DataFrame(Annotationframe.bbox.tolist(), index= Annotationframe.index)

  ### Make useful columns into a new data frame
  print(Annotationframe.head())
  print(Annotationframe.head())
  SmallFrame = Annotationframe[["id", "image_id","category_id","area","Xmin","Ymin","bboxWidth","bboxHeight"]]
  #### Make everything to one row per individual
  count = 0
  for y in Labelframe["name"]:
    temp = SmallFrame[SmallFrame["category_id"] == y] 
    temp.columns = ['id_'+ str(y), 'image_id', 'category_id_'+ str(y),'area_'+ str(y),'Xmin_'+ str(y),'Ymin_'+ str(y),'bboxWidth_'+ str(y),'bboxHeight_'+ str(y)]
    count += 1
    print(count)
    if count == 1:
      CompleteDataframe = temp
    else:
      CompleteDataframe = pd.merge(CompleteDataframe, temp, on = ["image_id"], how = "outer")

  CompleteDataframe.insert(3, 'Imagewidth', Imageframe['width']) 
  CompleteDataframe.insert(4, 'Imageheight', Imageframe['height']) 
  
  for x in range(len(CompleteDataframe["image_id"])):
    CompleteDataframe["image_id"][x] = CompleteDataframe["image_id"][x].split("/")[-1]
  ## save the data
 
  CompleteDataframe.to_csv(filename[:-5] + ".csv", index = False)
  
  return(CompleteDataframe,"Here ya go")


JsonToMeasurement(filename)




