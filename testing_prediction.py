import os
import imageio
import pandas as pd

# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt


# os.chdir("/home/anjonas/public/Public/Jonas/Data/ESWW006/ImagesNadir")
os.chdir('Z:/Public/Jonas/Data/ESWW006/ImagesNadir')
img_name = "2022_05_25/JPEG/20220525_Cam_ESWW0060022_Cnp_1.JPG"
image = imageio.imread(img_name)
base_name = os.path.basename(img_name)
stem_name = base_name.replace(".JPG", "")
coordinates = pd.read_csv(f"Meta/patch_coordinates/{stem_name}.txt")
c = coordinates.iloc[0, :].tolist()

patch = image[c[2]:c[3], c[0]:c[1]]





