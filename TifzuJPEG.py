import os as os
from PIL import Image


##### Convert all tifs to png. This step should be in a different document#####
path = "/Users/vg/Desktop/Daphnia_Raw/pulex"
current_path = path

for root, dirs, files in os.walk(current_path, topdown=False):
  print(dirs, files)
  for name in files:
    print(os.path.join(root, name))
    if os.path.splitext(os.path.join(root, name))[1].lower() == ".tif":
      if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".jpg"):
        print ("A jpeg file already exists for %s")
      else:
        outfile = os.path.splitext(os.path.join(root, name))[0] + ".jpg"
        try:
          im = Image.open(os.path.join(root, name))
          print ("Generating png for %s" )
          im.thumbnail(im.size)
          im.save(outfile, "JPEG", quality=100)
        except Exception:
          print("Except")
