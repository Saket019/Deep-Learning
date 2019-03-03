import os
import shutil
from PIL import Image 
import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models,transforms

from cifar import CIFAR10
from classifier_imageloader import ImageFolder
batch_size = 16 
transform = transforms.Compose(
	[
	transforms.Resize((224,224), interpolation=2),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	])

# trainset = ImageFolder("/home/saket/CVPRW/MVC/Mens/train", transform = transform, target_transform = None)
# trainloader = torch.utils.data.DataLoader(trainset, shuffle = True , batch_size = batch_size , num_workers = 4)
testset= ImageFolder("/home/saket/CVPRW/MVC/Mens/test", transform = transform, target_transform = None)
testloader = torch.utils.data.DataLoader(testset, shuffle = True , batch_size = int(batch_size/2) , num_workers = 4)
print("\nDataset generated. \n\n")

# trainset = CIFAR10("/home/saket/Saket/old_data/CIFAR/train",train =True,download = False,transform = transform)    
# trainloader = torch.utils.data.DataLoader(trainset, shuffle = True , batch_size = batch_size , num_workers = 2)
# testset = CIFAR10("/home/saket/Saket/old_data/CIFAR/test", train = False,download = False,transform = transform)
# testloader = torch.utils.data.DataLoader(testset, shuffle = True , batch_size = batch_size, num_workers = 2)

imgoutpath = "test.jpg"
imgoutpath2 = "test2.jpg"
for i,data in enumerate(testloader,0):
    input1, input2, labels, groundtruths1, groundtruths2 = data
    print(labels)
    combined = torch.cat((input1, input2), 0)
    torchvision.utils.save_image(combined, imgoutpath)
    break
