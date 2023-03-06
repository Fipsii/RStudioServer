### Masterfile from jpg to dataframe
import os
os.chdir('/home/philipp/GitRStudioServer/Workflow_code/')
from TifzuJPEG.py import ConvertTiftoJPEG
from Yaml_load_test.py import ConfigImport
from AnnotationRead.py import JsonToMeasurement
from DaphnidMeasure.py import DaphnidMeasurements
#from BodyWidth.py import RotatedImages, NonRightAngles, getOrientation, drawAxis, getDaphnidlength, DaphnidWidth
from DpiToMm.py import Images_list, getLineLength, get_Scale, NormalizeScale, makeDfwithfactors
from DataframeMerge.py import FuseDataframes, ShowMeasureImage


settings = ConfigImport("/home/philipp/GitRStudioServer/Workflow_code/settings.yml")
yaml = settings["Yaml_save_path"]
Working_folder = settings["Working_folder_path"]# /home/philipp/GitRStudioServer/ImageData/SimonaRelabelled/annotations/
Annotations = settings["Annotation_path"]# /home/philipp/GitRStudioServer/ImageData/SimonaRelabelled/annotations/instances_default.json
Images_folder = settings["Images_path"]# /home/philipp/GitRStudioServer/ImageData/SimonaRelabelled/images/SimonaAig21d/
scale_mode = settings["Scale_param"]

JsonToMeasurement(Annotations) ### Saves as Annotations[:-5] ".csv"
Measure = DaphnidMeasurements(Annotations[:-5] + ".csv") ### Saves as Annotations[:-5] ".csv"
Paths_of_Images, Name_of_Images = Images_list(Images_folder) ### Gives Image names and paths
Lines, CroppedImages = getLineLength(Paths_of_Images) ### Line lengths and lower right aprt of image
Units = get_Scale(CroppedImages, 200, 7) ## recognizes with tesseract ocr the Scale values
CleanUnits = NormalizeScale(Units) #### Makes one number or list out of list(list(n,n1,n2), list(n,n1,n2),...)
ScaleDataframe = makeDfwithfactors(CleanUnits,Lines, Name_of_Images, scale_mode) ### Calculate the factor mm per px

ScaleDataframe.to_csv(Save_path + "Scale.csv", index = False)

#RotatedCutImages = RotatedImages(Image_Names)
#line_cor = getDaphnidlength(RotatedCutImages)
#body_widths = DaphnidWidth(line_cor, RotatedCutImages)

DataFrame = FuseDataframes(Annotations[:-5] + ".csv", Save_path + "Scale.csv", Working_folder + "final_measure.csv")
#ShowMeasureImage(DataFrame,True)
