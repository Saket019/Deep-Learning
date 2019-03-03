import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
from torchvision import models,datasets,transforms
import os
import pickle
import operator
from PIL import Image



zsize = 32

def normalize(img):
    #print(img.shape)
    img = img.astype(np.float32)  # converting array of ints to floats
    img_a = img[0, :, :]
    img_b = img[1, :, :]
    img_c = img[2, :, :]  # Extracting single channels from 3 channel image
    # The above code could also be replaced with cv2.split(img) << which will return 3 numpy arrays (using opencv)

    # normalizing per channel data:
    img_a = (img_a - np.min(img_a)) / (np.max(img_a) - np.min(img_a))
    img_b = (img_b - np.min(img_b)) / (np.max(img_b) - np.min(img_b))
    img_c = (img_c - np.min(img_c)) / (np.max(img_c) - np.min(img_c))

    # putting the 3 channels back together:
    img_norm = np.empty(img.shape, dtype=np.float32)
    img_norm[0, :, :] = img_a
    img_norm[1, :, :] = img_b
    img_norm[2, :, :] = img_c
    return img_norm

def precision(q_class,ret_classes):
    initlist =  [int(q_class==i) for i in ret_classes]
    den = np.sum(initlist)
    if den==0:
        #print("NULL")
        return 0
    # print("initlist",initlist)
    x = 0
    preclist = [0]*len(initlist)
    for idx,pts in enumerate(initlist):
        x+=pts
        preclist[idx] = x/(idx+1)
    # print("preclist",preclist)
    num = np.dot(preclist,initlist)
    # print(num/den)
    return num/den

class Cifar_cauchy(nn.Module):
	def __init__(self):
		super(Cifar_cauchy,self).__init__()
		self.conv1 = nn.Conv2d(3, 64, 11, stride = 4, padding = 2)
		self.conv2 = nn.Conv2d(64, 192, 5, padding = 2)
		self.conv3 = nn.Conv2d(192, 384, 3, padding = 1)
		self.conv4 = nn.Conv2d(384, 256, 3, padding = 1)
		self.conv5 = nn.Conv2d(256, 256, 3, padding = 1)
		self.fc1 = nn.Linear(256 * 6 * 6, 4096)
		self.fc2 = nn.Linear(4096, 4096)
		self.fc3 = nn.Linear(4096, zsize)

	def forward(self,x):
		x = F.relu(self.conv1(x))
		x,indices1 = F.max_pool2d(x,(3,3),(2,2),return_indices = True)
		x = F.relu(self.conv2(x))
		x,indices2 = F.max_pool2d(x,(3,3),(2,2),return_indices = True)
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = F.relu(self.conv5(x))
		x,indices3 = F.max_pool2d(x,(3,3),(2,2),return_indices = True)
		x = x.view(x.size(0), 256 * 6 * 6)
		x = F.dropout(x)
		x = F.relu(self.fc1(x))
		x = F.dropout(x)
		x = F.relu(self.fc2(x))
		x = torch.tanh(self.fc3(x))
		return x

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder,self).__init__()
		self.conv1 = nn.Conv2d(3, 64, 11, stride = 4, padding = 2)
		self.conv2 = nn.Conv2d(64, 192, 5, padding = 2)
		self.conv3 = nn.Conv2d(192, 384, 3, padding = 1)
		self.conv4 = nn.Conv2d(384, 256, 3, padding = 1)
		self.conv5 = nn.Conv2d(256, 256, 3, padding = 1)
		self.fc1 = nn.Linear(256 * 6 * 6, 4096)
		self.fc2 = nn.Linear(4096, 4096)


	def forward(self,x):
		x = F.relu(self.conv1(x))
		x,indices1 = F.max_pool2d(x,(3,3),(2,2),return_indices = True)
		x = F.relu(self.conv2(x))
		x,indices2 = F.max_pool2d(x,(3,3),(2,2),return_indices = True)
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = F.relu(self.conv5(x))
		x,indices3 = F.max_pool2d(x,(3,3),(2,2),return_indices = True)
		x = x.view(x.size(0), 256 * 6 * 6)
		x = F.dropout(x)
		x = F.relu(self.fc1(x))
		x = F.dropout(x)
		x = self.fc2(x)
		return x,indices1,indices2,indices3
    
encoder = Encoder()

class CauchyNet(nn.Module):
    def __init__(self):
        super(CauchyNet,self).__init__()
        self.encoder = Encoder()
        self.fc3 = nn.Linear(4096, zsize)
    
    def forward(self,x):
        x,_,_,_ = self.encoder(x)
        x = self.fc3(F.relu(x))
        x = F.tanh(x)
        return x

# model = CauchyNet()
model = Cifar_cauchy()
print(model)

model_path = "/home/saket/CVPRW/log_vanilla_cauchy/encoder-51.pkl"
model.load_state_dict(torch.load(model_path))
model.cuda()


galleryfolderpath = "/home/saket/CVPRW/MVC/Mens/gallery"
queryfolderpath = "/home/saket/CVPRW/MVC/mens_query"
gallery = {}

print("\n\n Building Gallery .... \n")

with torch.no_grad():
    for img in os.listdir(galleryfolderpath):
        pil_im = Image.open(os.path.join(galleryfolderpath,img))
        pil_im = pil_im.resize((224,224))
        numpy_image = np.array(pil_im)
        numpy_image = numpy_image.transpose((2, 0, 1))
        numpy_image = normalize(numpy_image)
        numpy_image = np.array([numpy_image])
        torch_image = torch.from_numpy(numpy_image)
        torch_image = torch_image.type('torch.cuda.FloatTensor')
        
        h = model(torch_image)
        # print(h)
        # h = torch.ceil(h)[0]
        gallery[img] = h
        del(torch_image)

    print("\n Building Complete. \n")
    #query_image = "/home/saket/Saket/old_data/query/01_1_front26918.jpg"
    len_classes = 
    image_tensor = torch.Tensor(8*len_classes,3,224,224)
    flag = 0
    count = 0
    q_prec = 0
    for q_name in os.listdir(queryfolderpath):
        count+=1
        # print(count)
        q_class = q_name.split("-")[-1].split(".")[0]
        query_image = os.path.join(queryfolderpath,q_name)
        pil_im_q = Image.open(query_image)
        pil_im_q = pil_im_q.resize((224,224))
        numpy_image_q = np.array(pil_im_q)
        numpy_image_q = normalize(numpy_image_q.transpose((2, 0, 1)))
        numpy_image_q = np.array([numpy_image_q])
        torch_image_q  =torch.from_numpy(numpy_image_q)
        torch_image_q = torch_image_q.type("torch.cuda.FloatTensor")
        h_q = model(torch_image_q)
        # h_q = torch.ceil(h_q)[0]

        dist = {}
        for key in gallery.keys():
            h1 = gallery[key]
            h1norm = torch.div(h1,torch.norm(h1,p=2))
            h2norm = torch.div(h_q,torch.norm(h_q,p=2))
            dist[key] = torch.pow(torch.norm(h1norm - h2norm, p=2),2)*zsize/4
            # cs = torch.nn.functional.cosine_similarity(h1, h_q,dim = 0,eps=1e-8)
            # dist[key] = (1-cs)*zsize/2
            # if dist_cal <= 2:
            #     print(dist_cal) 
            # print(dist[key])
        sorted_pool = sorted(dist.items(), key=operator.itemgetter(1))[0:7]
        print("\n\n")
        print(q_name)
        print("\n\n")
        print(sorted_pool)
        # ret_classes = [sorted_pool[i][0].split("-")[-1].split(".")[0] for i in range(len(sorted_pool))]
        # q_prec += precision(q_class,ret_classes)

        image_tensor[flag] = torch_image_q
        for i,key in enumerate(sorted_pool):
            #print(key[0])
            img_path = os.path.join(galleryfolderpath,key[0])
            img = normalize(np.array(Image.open(img_path).resize((224,224))).transpose((2,0,1)))
            imgt = torch.from_numpy(img).type("torch.cuda.FloatTensor")
            #print(imgt)
            image_tensor[i+flag+1] = imgt
        flag+=8

    querynumber = len(os.listdir(queryfolderpath))
    print("Model"+ " ::  mAP :", q_prec/querynumber)
    storepath ='/home/saket/CVPRW/log_vanilla_cauchy/'
    imgoutpath = os.path.join(storepath, 'retoutput.png')
    if not os.path.exists(storepath):
        os.makedirs(storepath)
    torchvision.utils.save_image(image_tensor, imgoutpath)

    # image_tensor = torch.Tensor(16,3,224,224)
    # image_tensor[0] = torch_image_q
    # for i,key in enumerate(sorted_pool):
    #     print(key[0])
    #     img_path = os.path.join(galleryfolderpath,key[0])
    #     img = normalize(np.array(Image.open(img_path).resize((224,224))).transpose((2,0,1)))
    #     imgt = torch.from_numpy(img).type("torch.cuda.FloatTensor")
    #     #print(imgt)
    #     image_tensor[i+1] = imgt
    # retreival_path = "/home/saket/Saket/result/output3.png"
    # torchvision.utils.save_image(image_tensor, retreival_path)