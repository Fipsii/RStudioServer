##### Body width #####
######################

# General idea: 1. Rotate Daphnia straight 2. Cut out Daphnia and Calculate
# height 3. Measure from the side at height 2/3 of the Daphnia to most outer
# point of the Daphnid


####
def Images_list(path_to_images):
  import os as os
  Image_names = []
  for root, dirs, files in os.walk(path_to_images, topdown=False):
    print(dirs, files)
    for name in files:
      print(os.path.join(root, name))
      Image_names.append(os.path.join(root, name))
  return Image_names

def getOrientation(pts, img):

def AddtoDataFrame(BodyWidthValues)



### Data gets written in csv and calcualted in a last script makin all 
### cobversions (Right now(20.02.23) part of pxtomm.py but will be seperated)
### R_Script Simona evaluates the accuracy with chisq.test again
### How to differentiate between Daphnias that rotate down and ones that rotate up? 
### Eyes? If The eye has higher coordinates than the spina?

#### Versuchsraum
Image_Names = Images_list("/home/philipp/GitRStudioServer/ImageData/SimonaAig21d")
RotatedCutImages = RotatedImages(Image_Names)
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
  
    img = cv2.imread(name)
    nbg = remove(img)
    
  
    # Convert image to binary mask black and white
    gray = cv2.cvtColor(nbg, cv2.COLOR_BGRA2GRAY)
    binary_mask = gray.copy()
    binary_mask[binary_mask >0] = 255
    
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
    counting_list_of_errors.append(OnlyValue[0])
    ### How do we safe image
    
    ### Takes long we will just save the images
    ### Make list of images
    rotated_image_list.append(ndimage.rotate(binary_mask, 180-int(OnlyValue[0])))
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
      plt.imshow(ndimage.rotate(img, 180-int(rotation[0])))
      plt.show()
    elif len(liste) == 0:
      right_rotation = [0]
    else: 
      print("Liste hat mehr als einen Eintrag")
      right_rotation = [1]
    return right_rotation

def getOrientation(pts, img):
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
  print(cntr, "Cntr print")
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
  for x in StraightCutDaphnids:
    
    rows = np.copy(StraightCutDaphnids[:, ::50])
    sum_of_rows = np.sum(rows,axis=0)
