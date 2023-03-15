##### Body width #####
######################

# General idea: 1. Rotate Daphnia straight 2. Cut out Daphnia and Calculate
# height 3. Measure from the side at height 2/3 (from the bottom) of the Daphnia to most outer
# point of the Daphnid


####


### Data gets written in csv and calcualted in a last script makin all 
### cobversions (Right now(20.02.23) part of pxtomm.py but will be seperated)
### R_Script Simona evaluates the accuracy with chisq.test again
### How to differentiate between Daphnias that rotate down and ones that rotate up? 
### Eyes? If The eye has higher coordinates than the spina?

def Images_list(path_to_images):
  import os as os
  Image_names = []
  for root, dirs, files in os.walk(path_to_images, topdown=False):
    print(dirs, files)
    for name in files:
      print(os.path.join(root, name))
      Image_names.append(os.path.join(root, name))
  return Image_names

def RotatedImages(Imagenames):
  
  import numpy as np
  from math import atan2, cos, sin, sqrt, pi
  import cv2 as cv2
  from scipy import ndimage
  from rembg import remove
  import matplotlib.pyplot as plt
  
  ## This functions cuts out all background and rotates Daphnids based on the eigenvalues
  ## calculated. It draws contours to into binary masks of the cutout to find the middle.
  ## And then calculate and visualize the angle the Daphnid is lying. We sometimes get 
  ## Daphnids that are 180° into the false direction. We have 2 options find the correct
  ## side using the where the eggs may present a problem or we ingore it and calculate the
  ## body with from both sides and retroactively choose the right side (length?)
  rotated_image_list = []
  counting_list_of_errors = []
  for name in Image_Names:
    #print(name)
    img = cv2.imread(name)
    nbg = remove(img)
    #print(name)
  
    # Convert image to binary mask black and white
    gray = cv2.cvtColor(nbg, cv2.COLOR_BGRA2GRAY)
    binary_mask = gray.copy()
    binary_mask[binary_mask >50] = 255
    
    # Convert image to binary mask black and white fo eye detection
    # Problem: Eggs are also round and dark
    _, bw = cv2.threshold(gray,254, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) ## Based on this we 
    
    # Find all the contours in the thresholded image
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rotation = []
    
    for i, c in enumerate(contours):
    
      area = cv2.contourArea(c)
      if area < 100000:
        continue
      cv2.drawContours(img, contours, i, (0, 0, 255), 2)
      rotation.append(getOrientation(c,img)) 
  
    ### Discard values that are 90 or -90 degrees as they are straight or a 
    ### falsely detected part
    
    OnlyValue = NonRightAngles(rotation)
    #print(OnlyValue)
    counting_list_of_errors.append(OnlyValue[0])
    ### How do we safe image
    ### For later it will be beneficial to have a daphnid that is over 0 in every pixel
    ### That is why set everythin in the contours to 256
    color = [255, 255, 255]
    stencil = binary_mask
    cv2.fillPoly(stencil, contours, color)
    Filled = cv2.bitwise_and(binary_mask, stencil)
    
    ### Takes long we will just save the images
    ### Make list of images
    rotated_image_list.append(ndimage.rotate(Filled, 180-int(OnlyValue[0])))
  return(rotated_image_list)

def NonRightAngles(liste):
    import matplotlib.pyplot as plt
    right_rotation = []
    try:  
      liste.remove(90)
      liste.remove(-90)
    except: 
      pass
    if len(liste) == 1:
      right_rotation = [liste[0]]
    elif len(liste) == 0:
      right_rotation = [0]
    else: 
      print("Liste hat mehr als einen Eintrag")
      right_rotation = [1]
    return right_rotation

def getOrientation(pts, img):
  import cv2 as cv2
  import numpy as np
  from math import atan2, cos, sin, sqrt, pi
  ## [pca]
  # Construct a buffer used by the pca analysis
  sz = len(pts)
  data_pts = np.empty((sz, 2), dtype=np.float64)
  for i in range(data_pts.shape[0]):
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]
 
  # Perform PCA analysis
  mean = np.empty((0))
  mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
 
  # Store the center of the object
  cntr = (int(mean[0,0]), int(mean[0,1]))
  #print(cntr, "Cntr print")
  ## [pca]
 
  ## [visualization]
  # Draw the principal components
  cv2.circle(img, cntr, 3, (255, 0, 255), 2)
  p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
  p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
  drawAxis(img, cntr, p1, (255, 255, 0), 1)
  drawAxis(img, cntr, p2, (0, 0, 255), 5)
 
  angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
  ## [visualization]
 
  # Label with the rotation angle
  rotation_degree = -int(np.rad2deg(angle)) - 90
  label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
  textbox = cv2.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
  cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
  return rotation_degree

def drawAxis(img, p_, q_, color, scale):
  import cv2 as cv2
  import numpy as np
  from math import atan2, cos, sin, sqrt, pi
  
  p = list(p_)
  q = list(q_)
 
  ## [visualization1]
  angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
  hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 
  # Here we lengthen the arrow by a factor of scale
  q[0] = p[0] - scale * hypotenuse * cos(angle)
  q[1] = p[1] - scale * hypotenuse * sin(angle)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
 
  # create the arrow hooks
  p[0] = q[0] + 9 * cos(angle + pi / 4)
  p[1] = q[1] + 9 * sin(angle + pi / 4)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
 
  p[0] = q[0] + 9 * cos(angle - pi / 4)
  p[1] = q[1] + 9 * sin(angle - pi / 4)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
  ## [visualization1]
  
def getDaphnidlength(StraightCutDaphnids):
  ## We want to find the length of a Daphnid (body) to calculate the body width
  ## as we want to measure at 1/3 of the Distance counting from the head
  ## we count from both sides to avoid needing to detect the head
  ## take every 50th row and calcualted the amount of pixel that are non-zero?
  ## do we get problem with the eye? Binary mask has no eye!
  line_coordinates = []
  for x in range(0,len(RotatedCutImages)):
    import numpy as np
    ### Find the index row where the line is the longest
    ### 
    #print(x)
    columns = np.copy(RotatedCutImages[x][:, ::50]) ### take every 50th column
    #np.shape(RotatedCutImages[0][:, ::50])
    #np.shape(RotatedCutImages[0])
    sum_of_columns = np.sum(columns,axis=0)
    Longest_column, = np.where(sum_of_columns == np.amax(sum_of_columns))
    Index_of_the_Longest_column = int(Longest_column[0])*50 ## If mutliple same values exist take the first one
    
    ### Now we need to find the y coordinate where this line goes from 0 to some value
    ### Therefore we extract the line out of the image
    ### Somewhere we confuse columns and rows!!!!
  
    full_image = np.copy(RotatedCutImages[x])
    full_column = full_image[:,Index_of_the_Longest_column]
    
    #### No I want the index of the first number of 0 and the last over 0
    
    firstNonZero = np.argmax(full_column>0) ### Argmax stops at first true
    lastNonZero = len(full_column) - np.argmax(np.flip(full_column>0)) ## Flip to come from the other side
    
    #### Check if we have a continuus line but does it matter? Its a definition thing and dependent on how well rembg cuts out
    #### We have to test if making a binary mask may help exculding extremities while including the body -> species dependent
    
    line_coordinates.append((firstNonZero,lastNonZero))
  return line_coordinates
  #### Plot the line into the image

def DaphnidWidth(coordinates, RotatedCutImages):
  import numpy as np
  from matplotlib import pyplot as plt
  
  body_width_px = []
  
  for x in range(len(RotatedCutImages)):
    
    plt.clf()
    plt.imshow(RotatedCutImages[x])
    
    upper_third = int(coordinates[x][0] + (coordinates[x][1] - coordinates[x][0]) * 1/3)
    lower_third = int(coordinates[x][1] - (coordinates[x][1] - coordinates[x][0]) * 1/3)
    
    Thirds = (upper_third, lower_third)
    #### Get upper measurement ### We avoid a nested for loop for readability
    #### We measure from the outside on the upper third until we reach the daphnid
    #### We then want to find the midpoint of these two coordinates
    #### and measure from the inside
    #### We have to split the image on the centre point and measure the flipped left and
    #### normal right half and add them together
    #### This lets us avoid to detect extremities on the outside
    #######################################################################################
    body_width_temp = []
    
    for y in range(len(Thirds)):
      
      row = RotatedCutImages[x][Thirds[y],:] 
      row_Left = np.argmax(row > 100) ## get the start coordinate of the body on the left
      row_Right = len(row) - np.argmax(np.flip(row > 100)) ## get the end 
      
      ### find the halfway coordinate between left and Right
      # Cut the image into left and right along the middle of the outsides we found
      row_middle = round((row_Right + row_Left)/2)
      
      left_half = RotatedCutImages[x][:, :row_middle]
      right_half = RotatedCutImages[x][:, row_middle:]
      
      #### Find the values for the split daphnid
      
      row_middle_to_left = np.argmax(np.flip(left_half[Thirds[y],:]  == 0))
      row_middle_to_right = np.argmax(right_half[Thirds[y],:]  == 0)
      
      ### No we want to translate the the points back to our old image
      ### right coordinate would be the row_upper_third_middle_to_right
      ### + width of the left half
      
      coor_right = len(left_half[Thirds[y],:]) + row_middle_to_right
      
      ### the left coordinate the width of left box - row_upper_third_middle_to_right we found
      coor_left = len(left_half[Thirds[y],:]) - row_middle_to_left
      
      ### The length is the difference between these left and rigth
      body_width_temp.append(coor_right - coor_left) 
      
      plt.plot(coor_left, Thirds[y], coor_right, Thirds[y], marker = 'o', ls = '-')
      ######################################################################### right
    
    plt.show()  
    if body_width_temp[0] >= body_width_temp[1]:
      body_width_px.append(body_width_temp[0])
    else:
      body_width_px.append(body_width_temp[1])
      #### Draw a symbolic image
    from matplotlib import pyplot
    
  return body_width_px
    
    ## PRELIMINAIRY: We take the longer line as width. Optimally we would take the line
    ## closer to the eye (Furca movement). We also need to consider extremities incresing width
    ## as well as intestines being 0 values and fragments 
    ## All of this depends on rembg and preprocessing of the image -> we could set small islands of pixels 0
    ## Binary masks? An Oval containing mos of a Daphnid
    ## And how to measure width do we take most outer point or not?
    ## Even after that we have to check variance which results from the spina being included or not

Image_Names = Images_list("/home/philipp/GitRStudioServer/ImageData/SimonaAig21d")
RotatedCutImages = RotatedImages(Image_Names)
line_cor = getDaphnidlength(RotatedCutImages)
body_widths = DaphnidWidth(line_cor, RotatedCutImages)

import cv2 as cv2
from matplotlib import pyplot as plt
img_S = cv2.imread(Image_Names[98])
plt.clf()
plt.imshow(img_S)
plt.plot(body_widths)
plt.savefig("Test")

### Tasks left: Increase sharpness (maybe with different rembg model to exclude arms more) OR fit on Daphnid shape?
### Make width calculation dependent on Eye
### Make Body width calculation based on continuus measurment
### Doesn't work why


