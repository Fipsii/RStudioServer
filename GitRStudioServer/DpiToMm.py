### Goal: Read in an Image an autoamtically detect the Scale used:
## Caveats: Maybe we will need manual input of scale, 
##Do we perform this per Image or once provided the person doesn't change the zoom 

#def getDPI(Imagefolder):

import cv2 as cv2
import numpy as np
from PIL import Image

def getLineLength:
## Gaussiaun blur and image read ##
###################################

  img = cv2.imread("/home/philipp/SimonaMeasure.tif")
  imgWh = cv2.imread("/home/philipp/SimonaMeasureWhite.tif")
  
  height = imgWh.shape[0]
  width = imimgWhg.shape[1]
  
  cropped_image = imgWh[int(height*(3/4)):height,int(width*(3/4)):width]
  gray = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)
  
  kernel_size = 5
  blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
  
  low_threshold = 0
  high_threshold = 250
  edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
  
  #### Get lines 
  
  rho = 3  # distance resolution in pixels of the Hough grid
  theta = np.pi / 180  # angular resolution in radians of the Hough grid
  threshold = 20  # minimum number of votes (intersections in Hough grid cell)
  min_line_length = 200  # minimum number of pixels making up a line
  max_line_gap = 0  # maximum gap in pixels between connectable line segments
  line_image = np.copy(cropped_image) * 0  # creating a blank to draw lines on
  
  # Run Hough on edge detected image
  # Output "lines" is an array containing endpoints of detected line segments
  lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                      min_line_length, max_line_gap)
  
  for line in lines:
    for x1,y1,x2,y2 in line:
      cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
      
  lines_edges = cv2.addWeighted(cropped_image, 0.8, line_image, 1, 0)
  
  
  import matplotlib.pyplot as plt
  plt.clf()
  plt.imshow(line_image)
  plt.show()
  
  #### How do we get the right line?
  #### IDEA seperate all rows with values over 0. Then take the shortest one as the others would represent the frame
  #### Could also optimize to take frame and look into it
  #### Problem: we have more than one value for the middle row. Mean? Smallest? Biggest?
  
  Summe = np.sum(line_image[:,:,0], axis = 1) ## Takes the red values and sums every row
  SumNoZeros = Summe[Summe != 0] ### Drop all 0s from the frame
  PixelPerUnit = SumNoZeros.min()/255 ## take the min value 
  
  return(PixelPerUnit, cropped_image)

#### No we need to get the Number above the line
# def getUnit(cropped_Image) ### For py Tesseract we need sudo install

import pytesseract

mask = cv2.inRange(cropped_image, np.array([0, 0, 0]), np.array([200, 200, 200]))

krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
dlt = cv2.dilate(mask, krn, iterations=1)
thr = 255 - cv2.bitwise_and(dlt, mask)
number = pytesseract.image_to_string(thr, config="--psm 10")  

plt.imshow(mask)
plt.show() 

########### We also should detect mm or µm this can be achieved with pytesseract. but we also can guess as 
########### high values should always be µm and low ones mm

px/mm = PixelPerUnit/detectedNumber
