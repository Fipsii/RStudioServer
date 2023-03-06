### Goal: Read in an Image an autoamtically detect the Scale used:
## Caveats: Maybe we will need manual input of scale, 
##Do we perform this per Image or once provided the person doesn't change the zoom 


#### 
#### Read in all Images as names
def Images_list(path_to_images):
  import os as os
  PureNames = []
  Image_names = []
  for root, dirs, files in os.walk(path_to_images, topdown=False):
    print(dirs, files)
    for name in files:
      print(os.path.join(root, name))
      Image_names.append(os.path.join(root, name))
  for x in range(0,len(Image_names)):
    PureNames.append(Image_names[x].split("/")[-1])
  return Image_names, PureNames

def getLineLength(Image_names):
## Gaussiaun blur and image read ##
###################################
  import cv2 as cv2
  import numpy as np
  from PIL import Image
  import matplotlib.pyplot as plt
  
  list_of_lengths = []
  list_of_cropped_images = []
  
  for x in range(len(Image_names)):
    img = cv2.imread(Image_names[x])
    print(x, "Was da los", Image_names[x])
    height = img.shape[0]
    width = img.shape[1]
    cropped_image = img[int(height*(3/4)):height,int(width*(3/4)):width]
    gray = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)
    list_of_cropped_images.append(gray)
    
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    
    low_threshold = 100
    high_threshold = 200
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    
    #### Get lines 
    
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 200 # minimum number of pixels making up a line
    max_line_gap = 0  # maximum gap in pixels between connectable line segments
    line_image = np.copy(cropped_image) * 0  # creating a blank to draw lines on
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
    
    if lines is None:
      list_of_lengths.append(0) 
    else:
      for line in lines:
        for x1,y1,x2,y2 in line:
          cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    
      lines_edges = cv2.addWeighted(cropped_image, 0.8, line_image, 1, 0)
      #plt.clf()
      #plt.imshow(lines_edges)
      #plt.show()
   
      #### How do we get the right line?
      #### IDEA seperate all rows with values over 0. Then take the shortest one as the others would represent the frame
      #### Could also optimize to take frame and look into it
      #### Problem: we have more than one value for the middle row. Mean? Smallest? Biggest?
      
      Summe = np.sum(line_image[:,:,0], axis = 1) ## Takes the red values and sums every row
      SumNoZeros = Summe[Summe != 0] ### Drop all 0s from the frame
      ### To prevent a non existing value changing the lenght of a list (like missing scale we set pixelper Unit to 0)
      ### and then assign it the real value if existent
      PixelPerUnit = SumNoZeros.min()/255 ## take the min value 
      list_of_lengths.append(PixelPerUnit) 
      plt.clf()
      plt.imshow(line_image)
      plt.show()
  return(list_of_lengths, list_of_cropped_images)

#### No we need to get the Number above the line
# def getUnit(cropped_Image) ### For py Tesseract we need sudo install
### white only needs mask
### We need to check if the scale is white
### Therefore we check if we have more 0 pixels in the mask than 255
def get_Scale(cropped_images, in_Range_upper_limit, psm_mode):
  import pytesseract as pytesseract
  import matplotlib.pyplot as plt
  import cv2 as cv2
  ScaleUnit = []
  for x in range(len(CroppedImages)):

    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

    ## Crop the iamge further to get clearer Image 
    height = CroppedImages[x].shape[0]
    width = CroppedImages[x].shape[1]
    CroppedImagess = CroppedImages[x][int(height*0.75):height,0:int(width)]
    
    # For white scales that have almost the same background 
    ContrastforWhiteScales = CroppedImagess-250 ## This increases contrast for white scales
    ContrastforWhiteScalesA = cv2.adaptiveThreshold(ContrastforWhiteScales,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,15) ### Normalize
    plt.clf()
    plt.imshow(cv2.cvtColor(ContrastforWhiteScalesA, cv2.COLOR_BGR2RGB))
    plt.show()
  
    ## Scales that already have contrast
    MaskCorrNorm = cv2.normalize(CroppedImagess,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)  #### Normalize
    MaskCorrNormBare = cv2.inRange(MaskCorrNorm,0,255) ## This works very well for 500 µm but very badly for 1mm
    MaskCorrNormBare = cv2.bitwise_and(MaskCorrNorm,MaskCorrNorm, mask= MaskCorrNormBare) ####  Use mask to make 0/1 mImage
    MaskCorrNormBareB = cv2.GaussianBlur(MaskCorrNormBare, (3,3), sigmaX=70, sigmaY=70) ## blur for divide increases sharpness maybe not necessary
    MaskCorrNormBareV = cv2.divide(MaskCorrNormBare, MaskCorrNormBareB, scale=255)
    Inverse = cv2.bitwise_not(MaskCorrNormBareV) ## Make inverse as black letters are easier to detecet
    
    ## tesseract detect
    psm_mode = 7
    tesseract_config = r'--oem 3 --psm ' + str(psm_mode) +" -c tessedit_char_whitelist=0123456789"
    
    ## as we dont no which kind of scale was put in we use all 3 (inverse, contrast for white and MasCorrNormBare) and then compare.
    ## We could find out the colour of the scale when we use the lines, but is it necessary?
    
    number = pytesseract.image_to_string(MaskCorrNormBare, config= tesseract_config).strip()
    number2 = pytesseract.image_to_string(Inverse, config= tesseract_config).strip()
    number3 = pytesseract.image_to_string(ContrastforWhiteScalesA, config= tesseract_config).strip()
    ScaleUnit.append([number,number2,number3]) 
  return(ScaleUnit)

########### We also should detect mm or µm this can be achieved with pytesseract. but we also can guess as 
########### high values should always be µm and low ones mm ## psm 7 is good by far not perfect... Run multiple modes?

def NormalizeScale(ReadInsScale): ### Make all values into mm and decide which value is true and which not
  #### We will drop 0 values out of the list and convert 7 into 1s as nobody has a 7 as scale
  #### We then compare if its the only value take it
  #### If we have mutliple same take that
  #### If we have multiple but different values: Ask user? have a list of likely numbers and throw a warning? likely numbers: 1,2,100,200,250,300,400,500,600,700,750,800,900,1000,5
  #### If we have multiple likely numbers? ask USer? 
  likely_numbers = ["1","2","100","200","250","300","400","500","600","700","750","800","900","1000","5"]
  ScaleUnitsClear = []
  str_list = []
  for x in range(len(Units)): ## for all
      str_list.append(list(filter(None, Units[x]))) ### Drop all empty entries
  
  for x in range(len(str_list)-1): ## for al entries
    for y in str_list[x]: ### if all values in every list entry
      print(x,y)
      if y in likely_numbers: ### if a value is in the likely numbers list
        str_list[x] = int(y) ## make the entry the number
  a =str_list.copy()
  str_list = a
  ### Delete uncertainties and set them to 0

  for x in range(len(str_list)):
    if type(str_list[x]) == list:
     str_list[x] = float("nan")
  ### Now we need to make every value above 50/1000
  for x in range(len(str_list)):
    if str_list[x] > 49:
      str_list[x] = str_list[x]/1000
  return(str_list)  
  
def makeDfwithfactors(ScaleUnitClean, list_of_lengths, list_of_names, One_scale, ConvFactor=0.0028):
  ### This function has two modes. 1) If the user declares taht we only have one 
  ### scale we take the most common values of length and unit and 2) if more 
  ### than one exist we keep the list as they are.
  ### Then we enter the singular or mutliple values into the df
  import pandas as pd
  LengthOpt = [int(item) for item in list_of_lengths] ## Make linelengths int
  if One_scale == 0:
    Scale_df = pd.DataFrame(list_of_names, columns =['Name']) 
    Scale_df["distance_per_pixel"] = ConvFactor
    print("Using manual factor of Leica Olympus 0.028 px/mm")
    return Scale_df

  elif One_scale > 0:
    
    if One_scale == 1:
      LengthOpt = max(set(list_of_lengths), key = list_of_lengths.count)
      UnitOpt = max(set(ScaleUnitClean),key = ScaleUnitClean.count)
      
    else:
	    LengthOpt = list_of_lengths
	    UnitOpt = ScaleUnitClean
    
    Scale_df = pd.DataFrame(list_of_names, columns =['Name'])
    print(list_of_lengths)
    Scale_df["metric_length"] = UnitOpt
    Scale_df["scale[px]"] = LengthOpt
    Scale_df["distance_per_pixel"] = Scale_df["metric_length"]/Scale_df["scale[px]"]
    return Scale_df

##### Testing space for image prep
##225 or 468
#### Make Lines to Int ### Do we make a config for all Codes?
### 0 = no scale native resolution of 2.8 px/µm with standard zoom (If you know the configuration of
### your own microscope enter it into the config.cfg)
### 1 = one scale for all images (automatically detects scale and Unit for every
### image but takes the most common value for all images -> robust)
### 2 = Multiple scales: tries to find a scale for every image. May be prone to
### error

from Yaml_load_test import ConfigImport

settings = ConfigImport("/home/philipp/GitRStudioServer/Workflow_code/settings.yml")
scale_cfg = settings["Scale_param"]
path = settings["Images_path"]
Save_path = settings["Working_folder_path"]

Paths_of_Images, Name_of_Images = Images_list(path)
Lines, CroppedImages = getLineLength(Paths_of_Images)
Units = get_Scale(CroppedImages, 200, 7)
CleanUnits = NormalizeScale(Units)
ScaleDataframe = makeDfwithfactors(CleanUnits,Lines, Name_of_Images, scale_cfg)

ScaleDataframe.to_csv(Save_path + "Scale.csv", index = False)
