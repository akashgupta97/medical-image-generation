import os
import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space
from glob import glob
from PIL import Image as im

DCMs = glob('*.dcm')

count=0
for dcm in DCMs:
    if count%1000==0:
        print(count)
    count+=1
    split_path = dcm.split("/")[2:]
    split_path[-1] = split_path[-1].split(".dcm")[0] + ".png"
    sav_path = "data/train_val_preprocessed/" + "/".join(split_path)
    ds = pydicom.dcmread(dcm)
    try:
        is_color = ds.SamplesPerPixel == 3
    except Exception:
        is_color = False
    try:
        color_space = ds.PhotometricInterpretation
    except Exception:
        color_space = ""

    arr = ds.pixel_array
    if is_color and color_space != "RGB":
        arr = convert_color_space(arr, color_space, "RGB")
    data = im.fromarray(arr)
    os.makedirs("/".join(sav_path.split("/")[:-1]))
    data.save(sav_path)