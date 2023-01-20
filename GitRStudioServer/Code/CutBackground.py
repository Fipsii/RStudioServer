import rembg

from PIL import Image
import keras


Magna = load_img("/Users/vg/Desktop/Daphnia/magna/DeTaVw_softver thingy_82.png", color_mode = "rgb")
Pulex = load_img("/Users/vg/Desktop/Daphnia/pulex/Po12_189.png", color_mode = "rgb")
Cucullata = load_img("/Users/vg/Desktop/Daphnia/cucullata/D.cucullata_2017-07-18_I1_W (1).png", color_mode = "rgb")
Longhicephala = load_img("/Users/vg/Desktop/Daphnia/longicephala/C-13.png", color_mode = "rgb")

InputImg = Image.open(Magna)
Without_background = rembg.remove(InputImg)
Image.open(InputImg)
conda list
