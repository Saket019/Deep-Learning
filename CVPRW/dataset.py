import numpy as np 
import pandas as pd
import os
import random
import itertools

def dataset(fpath):
    # fpath = "/home/saket/CVPRW/MVC/Mens/train"
    if not os.path.exists(fpath):
        raise Exception("Invalid path destination")
    if fpath.split("/")[-1]=="train":
        im_thresh = 5000
        type2_num = 200000
        type1_num = 100000
    elif fpath.split("/")[-1]=="test":
        im_thresh = 1500
        type1_num = 10000
        type2_num = 10000
 
    classes = [d for d in os.listdir(fpath)]
    classes = sorted(classes)
    cls_to_idx = {classes[i]:i for i in range(len(classes))}
    # print(classes)
    # print(cls_to_idx)
    cls_num_imgs = {cls_to_idx[i]:len(os.listdir(os.path.join(fpath,i))) for i in classes}
    # print(cls_num_imgs)
    images1 = []
    images2 = []
    dataset2 = []
    for cls_name in classes:
        images_temp = [(im,cls_name) for im in os.listdir(os.path.join(fpath,cls_name))]
        random.shuffle(images_temp)
        images1.extend(images_temp[:200])
        images2 = images_temp[:min(im_thresh,len(images_temp))]
        # images3 = images_temp[200:min(500,len(images_temp))]
        dataset2.extend(list(itertools.combinations(images2,2)))    

    # print(len(images1))
    dataset1 = list(itertools.combinations(images1,2))
    # print(dataset1[0])
    # print(len(dataset1))
    # print(len(dataset2))


    count = [0]*3
    newdataset = []
    for data in dataset1:
        if not data[0][1] == data[1][1] :
            t = 2
            count[2]+=1
            img_path1 = os.path.join(os.path.join(fpath,data[0][1]), data[0][0])
            img_path2 = os.path.join(os.path.join(fpath,data[1][1]), data[1][0])
            item = (img_path1,img_path2,t)
            newdataset.append(item)
            if count[2] == type2_num:
                break

    random.shuffle(dataset2)
    for data in dataset2:
        img_path1 = os.path.join(os.path.join(fpath,data[0][1]), data[0][0])
        img_path2 = os.path.join(os.path.join(fpath,data[1][1]), data[1][0])    
        if data[0][0].split("-")[0] == data[1][0].split("-")[0]:
            t = 0
            count[0]+=1
            item = (img_path1,img_path2,t)
            # item = (data[0][0],data[1][0],t)
            newdataset.append(item)
        else:
            t = 1
            if count[1] < type1_num :
                count[1]+=1
                item = (img_path1,img_path2,t)
                newdataset.append(item)
        

    random.shuffle(newdataset)
    # print("Count",count)
    # print(newdataset[:5])
    return newdataset


"""
#data csv creation

fpath = "/home/siplab/Saket/old_data/train"
dataname ={}

for f in os.listdir(fpath):
    print(f)
    dataname[f] = []
    classpath = os.path.join(fpath,f)
    for g in os.listdir(classpath):
        dataname[f].append(os.path.join(classpath,g))
    print(len(dataname[f]))

df = pd.DataFrame(columns=["image1","image2","s"])
for key in dataname.keys():
    print(key)
    for i in range(min(50,len(dataname[key])-2)):
        path1 = dataname[key][i]
        for j in range(i,min(50,len(dataname[key])-1)):
            path2 = dataname[key][j+1]
            df = df.append({"image1":path1, "image2":path2, "s":0}, ignore_index= True)
            #print(df)
            #print(path1,path2)
file_name = "/home/siplab/Saket/old_data/similar50.csv"
df.to_csv(file_name) #, sep='\t', encoding='utf-8')
"""