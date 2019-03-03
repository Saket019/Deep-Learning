import os
import shutil
import numpy as np 

split_ratio = 0.4 #test:train

fpath = "/home/saket/CVPRW/Mens"
testpath = "/home/saket/CVPRW/test"
gallerypath = "/home/saket/CVPRW/gallery"
if not os.path.exists(gallerypath):
    os.makedirs(gallerypath)
querypath = "/home/saket/CVPRW/query"
if not os.path.exists(querypath):
    os.makedirs(querypath)


if not os.path.exists(fpath):
    raise Exception("Invalid path destination")
classes = [d for d in os.listdir(fpath)]
classes = sorted(classes)
cls_to_idx = {classes[i]:i for i in range(len(classes))}
cls_num_imgs = {cls_to_idx[i]:len(os.listdir(os.path.join(fpath,i))) for i in classes}
print(cls_num_imgs)

for clsname in classes:
    clspath = os.path.join(fpath,clsname)
    destpath = os.path.join(testpath,clsname)
    if not os.path.exists(destpath):
        os.makedirs(destpath)
    images = []
    for img in os.listdir(clspath):
        images.append(img)
    images = sorted(images)
    for i in range(int(split_ratio*len(images))):
        if i <= 0.2*len(images):       
            shutil.move(os.path.join(clspath,images[i]),os.path.join(destpath,images[i]))
        else:
            imgname= "-".join(images[i].split("-")[:2])+"-"+clsname + ".jpg"
            # print(imgname)
            if images[i].split("-")[1] == "p":
                shutil.move(os.path.join(clspath,images[i]),os.path.join(querypath,imgname))
            else:
                shutil.move(os.path.join(clspath,images[i]),os.path.join(gallerypath,imgname))


