### Goal: Read in an Image an autoamtically detect the Scale used:
## Caveats: Maybe we will need manual input of scale, 
##Do we perform this per Image or once provided the person doesn't change the zoom 
import os as os
path = "/home/philipp/GitRStudioServer/ImageData/SimonaAig21d"
#### 
#### Read in all Images as names
def Images_list(path_to_images):
  Image_names = []
  for root, dirs, files in os.walk(path, topdown=False):
    print(dirs, files)
    for name in files:
      print(os.path.join(root, name))
      Image_names.append(os.path.join(root, name))
  return Image_names

def getLineLength(Image_names):
## Gaussiaun blur and image read ##
###################################
  import cv2 as cv2
  import numpy as np
  from PIL import Image
  import matplotlib.pyplot as plt
  
  list_of_lengths = []
  list_of_cropped_images = []
  
  for x in range(len(Image_names)-1):
    img = cv2.imread(Image_names[x])
    height = img.shape[0]
    width = img.shape[1]
    
    cropped_image = img[int(height*(3/4)):height,int(width*(3/4)):width]
    gray = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)
    list_of_cropped_images.append(gray)
    
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    
    low_threshold = 100
    high_threshold = 200
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    #### Get lines 
    
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 100  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 200 # minimum number of pixels making up a line
    max_line_gap = 0  # maximum gap in pixels between connectable line segments
    line_image = np.copy(cropped_image) * 0  # creating a blank to draw lines on
    type(line_image)
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    plt.clf()
    plt.imshow(gray)
    plt.show()
    plt.clf()
    plt.imshow(line_image)
    plt.show()
    
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
    print(type(lines))
    if lines is None:
      list_of_lengths.append("NoScaledetected_check if one exists") 
    else:
      for line in lines:
        for x1,y1,x2,y2 in line:
          cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    
      lines_edges = cv2.addWeighted(cropped_image, 0.8, line_image, 1, 0)
  
    
   
    #### How do we get the right line?
    #### IDEA seperate all rows with values over 0. Then take the shortest one as the others would represent the frame
    #### Could also optimize to take frame and look into it
    #### Problem: we have more than one value for the middle row. Mean? Smallest? Biggest?
      print(Image_names[x+1])
      Summe = np.sum(line_image[:,:,0], axis = 1) ## Takes the red values and sums every row
      SumNoZeros = Summe[Summe != 0] ### Drop all 0s from the frame
      PixelPerUnit = SumNoZeros.min()/255 ## take the min value 
      list_of_lengths.append(PixelPerUnit) 
  return(list_of_lengths, list_of_cropped_images)

#### No we need to get the Number above the line
# def getUnit(cropped_Image) ### For py Tesseract we need sudo install
### white only needs mask
### We need to check if the scale is white
### Therefore we check if we have more 0 pixels in the mask than 255
def get_Scale(cropped_images, in_Range_upper_limit, psm_mode):

  import pytesseract
  ScaleUnit = []
  for x in range(len(cropped_images)):
    
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    
    ### Make mask and inverse mask for white and black scales 
    ### Yellow background black characters
    
    mask = cv2.inRange(cropped_images[x], 0, in_Range_upper_limit)
    #blur = cv2.GaussianBlur(mask, (0,0), sigmaX=33, sigmaY=33)
    #blur = cv2.medianBlur(mask, 1)
    blur = cv2.bilateralFilter(mask,9,75,75)
    # divide
    divide = cv2.divide(mask, blur, scale=255)
    Inverse = cv2.bitwise_not(mask)
  
    #### make the brighter image the analysis mask
    if sum(sum(Inverse)) < sum(sum(divide)):
      maskCorr = Inverse
        
    else:
      maskCorr = divide
      
    plt.clf()
    plt.imshow(maskCorr)
    plt.show()
    tesseract_config = r'--oem 3 --psm ' + str(psm_mode) +" -c tessedit_char_whitelist=0123456789"
    number = pytesseract.image_to_string(maskCorr, config= tesseract_config).strip()
    ScaleUnit.append(number)
  return(ScaleUnit)
########### We also should detect mm or µm this can be achieved with pytesseract. but we also can guess as 
########### high values should always be µm and low ones mm ## psm 7 is good by far not perfect... Run multiple modes?

Images = Images_list(path)
range(len(Images))
Lines, CroppedImages = getLineLength(Images)
Units = get_Scale(CroppedImages, 200, 7)
print(Units)
Units[0]


##### Testing space for image prep
ScaleUnit = []
for x in range(len(CroppedImages)):
    
  pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    
  ### Make mask and inverse mask for white and black scales 
  ### Yellow background black characters
 
  height = CroppedImages[x].shape[0]
  width = CroppedImages[x].shape[1]
  CroppedImagess = CroppedImages[x][int(height*0.75):height,0:int(width)]
  
  # For scales that have almost the same background
  ContrastforWhiteScales = CroppedImagess-250
  #NormalizedWhiteScale = cv2.normalize(ContrastforWhiteScales,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
  #WhitetoblackScale = cv2.inRange(NormalizedWhiteScale,150,255) ## This works very well for 500 µm but very badly for 1mm
  #BlackAndWhite = cv2.bitwise_and(NormalizedWhiteScale,NormalizedWhiteScale, mask= WhitetoblackScale)
  #BlurMask = cv2.GaussianBlur(BlackAndWhite, (3,3), sigmaX=70, sigmaY=70)
  #ScaleBlackfromWhite = cv2.divide(BlackAndWhite, BlurMask, scale=255)
  ContrastforWhiteScalesA = cv2.adaptiveThreshold(ContrastforWhiteScales,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,15)
  plt.clf()
  plt.imshow(cv2.cvtColor(ContrastforWhiteScalesA, cv2.COLOR_BGR2RGB))
  plt.show()
  
  ## Scales that already have contrast
  MaskCorrNorm = cv2.normalize(CroppedImagess,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
  MaskCorrNormBare = cv2.inRange(MaskCorrNorm,0,255) ## This works very well for 500 µm but very badly for 1mm
  MaskCorrNormBare = cv2.bitwise_and(MaskCorrNorm,MaskCorrNorm, mask= MaskCorrNormBare)
  MaskCorrNormBareB = cv2.GaussianBlur(MaskCorrNormBare, (3,3), sigmaX=70, sigmaY=70)
  MaskCorrNormBare = cv2.divide(MaskCorrNormBare, MaskCorrNormBareB, scale=255)
  #MaskCorrNormBare = MaskCorrNorm*15
  #MaskCorrNormBare = MaskCorrNormBare.astype(int)
  Inverse = cv2.bitwise_not(MaskCorrNormBare)
  plt.clf()
  plt.imshow(cv2.cvtColor(ContrastforWhiteScalesA, cv2.COLOR_BGR2RGB))
  plt.show()
  psm_mode = 7
  tesseract_config = r'--oem 3 --psm ' + str(psm_mode) +" -c tessedit_char_whitelist=0123456789"
  number = pytesseract.image_to_string(MaskCorrNormBare, config= tesseract_config).strip()
  number2 = pytesseract.image_to_string(Inverse, config= tesseract_config).strip()
  number3 = pytesseract.image_to_string(ContrastforWhiteScalesA, config= tesseract_config).strip()
  ScaleUnit.append([number,number2,number3])

len(ScaleUnit)

#### This Works now we have to sort out the doubles and false positives (mostly 0s that are easy to discard) ### We also have to discuss the few false line values in the second funtion
