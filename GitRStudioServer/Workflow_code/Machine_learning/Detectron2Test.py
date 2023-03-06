#### Detectron2 Evaluation of Ginjinn output ####
import detectron2 
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
import cv2 as cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog 
###
im = cv2.imread("SimonaMeasure.tif")
### Load model ###
cfg = detectron2.config.get_cfg()
cfg.merge_from_file(detectron2.model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))
cfg.MODEL.DEVICE = "cpu" 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
cfg.MODEL.WEIGHTS = '/home/philipp/daphnia_project3.0/outputs/model_final.pth' # Set path model

#ginjinn_config.yaml
### Evaluate with data ###
predictor = detectron2.engine.DefaultPredictor(cfg)
outputs = predictor(im)

print(outputs["instances"].pred_boxes)
### Something is wrong do not know what have to learn pytorch for that D:
# We can use `Visualizer` to draw the predictions on the image.
v =  Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

from matplotlib import pyplot as plt

plt.imsave("DetectronTest.jpg",out.get_image()[:, :, ::-1])

### PROBLEM: GinJinn uses custom functions that are not easily implemented... What do?

#### Get GinJinn config
cls
gj_cfg
img_dir
outdir
checkpoint_name = "model_final.pth"

d2_cfg = gj_cfg.to_detectron2_config()
d2_cfg.MODEL.WEIGHTS = os.path.join(d2_cfg.OUTPUT_DIR, checkpoint_name)
def from_ginjinn_config(
        cls,
        gj_cfg: GinjinnConfiguration,
        img_dir: str,
        outdir: str,
        checkpoint_name: str = "model_final.pth",
    ) -> "GinjinnPredictor":
        """
        Build GinjinnPredictor object from GinjinnConfiguration instead of
        Detectron2 configuration.

        Parameters
        ----------
        gj_cfg : GinjinnConfiguration
        img_dir : str
            Directory containing input images for inference
        outdir : str
            Directory for writing results
        checkpoint_name : str
            Name of the checkpoint to use.

        Returns
        -------
        GinjinnPredictor
        """

        d2_cfg = gj_cfg.to_detectron2_config()
        d2_cfg.MODEL.WEIGHTS = os.path.join(d2_cfg.OUTPUT_DIR, checkpoint_name)

        return cls(
            d2_cfg, get_class_names(gj_cfg.project_dir), img_dir, outdir, gj_cfg.task
        )
