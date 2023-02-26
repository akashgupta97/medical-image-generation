from scipy import ndimage
import imageio
import matplotlib.pyplot as plt
from glob import glob
import os

PNGs = glob("data/train_val_preprocessed/*/*/*.png")

count=0
for png in PNGs:
    count+=1
    sav_path = "data/conditions/laplace_gaussian/" + "/".join(png.split("/")[2:])
    xray_image = imageio.v3.imread(png)
    xray_image_laplace_gaussian = ndimage.gaussian_laplace(xray_image, sigma=1)
    fig, axes = plt.subplots()
    im = axes.imshow(xray_image_laplace_gaussian)
    axes.set_axis_off()
    os.makedirs("/".join(sav_path.split("/")[:-1]))
    fig.savefig(sav_path,  bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    if count%5000 == 0:
        print(count)
