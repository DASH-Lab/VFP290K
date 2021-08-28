import cv2
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
import os
from os.path import join as opj

import xmltodict
from tqdm.notebook import tqdm

class args:
    data_root_dir = "/media/data1/VFP290K"
    mosaic_ratio = 5
    video_name = "*"


label_paths = sorted(glob(f"{args.data_root_dir}/{args.video_name}/clean_xml/*.xml"))
img_paths = [x.replace("VFP_release2", "VFP_release").replace("xmls_overlap95_nocar", "images").replace(".xml", ".jpg").replace("_done", "") for x in label_paths]

for img_path, label_path in tqdm(zip(img_paths, label_paths)):
    try:
        img = cv2.imread(img_path)

        with open(label_path, 'r') as f:
            xml = xmltodict.parse(f.read())

        car_box = []

        if "object" in xml["annotation"].keys():
            obj = xml["annotation"]["object"]
            if type(obj) is not list:  # object가 1개
                name = obj["name"]
                if name == "2":
                    car_box.append(obj["bndbox"])

            else: 

                for ob in obj:
                    name = ob["name"]
                    if name == "2":
                        car_box.append(ob["bndbox"])

        else:  # no object
            pass


        if len(car_box) != 0:
            for bnd_box in car_box:
                x_min = int(bnd_box["xmin"]) 
                x_max = int(bnd_box["xmax"]) 
                y_min = int(bnd_box["ymin"])
                y_max = int(bnd_box["ymax"]) 
                w = x_max - x_min
                h = y_max - y_min

                patch = img[y_min:y_max, x_min:x_max]
                for _ in range(6):
                    mosaic = cv2.resize(patch, (w//args.mosaic_ratio, h//args.mosaic_ratio))
                    mosaic = cv2.resize(mosaic, (w,h))
                img[y_min:y_max, x_min:x_max] = mosaic   

        dir_, fn = os.path.split(img_path)
        to_dir = dir_.replace("GOPR", "2_GOPR")
        os.makedirs(to_dir, exist_ok=True)
        to_path = opj(to_dir, fn)
        cv2.imwrite(to_path, img)
    except:
        print(label_path)  # manual mosaic
