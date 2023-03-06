
from Yaml_load_test import ConfigImport

settings = ConfigImport("/home/philipp/GitRStudioServer/Workflow_code/settings.yml")

##### Convert all non jpg to jpg, Ignores files that cant be made to jpg. 

def ConvertTiftoJPEG(directory):
  import os as os
  from PIL import Image
  for root, dirs, files in os.walk(directory, topdown=False):
    for name in files:
      if os.path.splitext(os.path.join(root, name))[1].lower() != ".jpg":
        if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".jpg"):
          print ("A jpeg file already exists for %s")
        else:
          outfile = os.path.splitext(os.path.join(root, name))[0] + ".jpg"
          try:
            im = Image.open(os.path.join(root, name))
            print ("Generating jpg for %s" )
            im.thumbnail(im.size)
            im.save(outfile, "JPEG", quality=100)
          except Exception:
            print("Exception occured. If yoot folder only contains images ignore this message")

ConvertTiftoJPEG(settings["Images_path"])
