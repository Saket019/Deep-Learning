import numpy as np
import json
import shutil
import requests
import os

file = "/home/saket/Downloads/mvc_info.json"
with open(file) as f:
    data = json.load(f)

for i in range(len(data)):
    url = data[i]['image_url_multiView']    

    classname = data[i]['category'].split('"')[1]
    subclass = data[i]['subCategory2'].split('"')[1]
    save_img_name = url.split("/")[-1]
    gender =  data[i]['productGender'].split('"')[1]
    
    superfolder = os.path.join("/home/saket/CVPRW2/MVC", gender)
    if not os.path.exists(superfolder):
        os.makedirs(superfolder)

    folder = os.path.join(superfolder, classname)
    if not os.path.exists(folder):
        os.makedirs(folder)

    subfolder = os.path.join(folder, subclass)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)    

    img_path = os.path.join(subfolder, save_img_name)
    if not os.path.exists(img_path):
        try :
            response = requests.get(url, stream=True)
        except requests.exceptions.MissingSchema:
            continue
        with open(img_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response
