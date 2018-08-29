import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models,transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np 
import os
import matplotlib.pyplot as plt 
from torch.autograd import Function
from random import randint
import operator

batch_size = 1
zsize = 48
model = '/home/siplab/Saket/fashion/Encoder/encoder-18.pkl'

transform = transforms.Compose(
	[
	transforms.Scale((224,224), interpolation=2),
	transforms.ToTensor(),
	])

trainset=torchvision.datasets.ImageFolder("/home/siplab/Saket/fashion/Img/dataset/train", transform=transform, target_transform=None)
trainloader = torch.utils.data.DataLoader(trainset, shuffle = True , batch_size = batch_size , num_workers = 2)
testset=torchvision.datasets.ImageFolder("//home/siplab/Saket/fashion/Img/dataset/test", transform=transform, target_transform=None)
testloader = torch.utils.data.DataLoader(testset, shuffle = True , batch_size = batch_size , num_workers = 2)


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
		x = self.fc3(x)
		x = F.sigmoid(x)		
		return x   #,indices1,indices2,indices3
	
encoder = Encoder()
encoder.cuda()
encoder.load_state_dict(torch.load(model,map_location=lambda storage, loc: storage.cuda(1)))
#encoder.load_state_dict(torch.load(model,map_location={'cuda:1':'cuda:2'}))
#encoder = torch.nn.DataParallel(encoder,device_ids=[1,2])
#print encoder 
binary_test_dict = {}
binary_train_dict = {}
testdict = {} 
f7 = {}
labeldict = {}
traindict={}

print "Creating Dictionaries"
for i,data in enumerate(testloader):	
	if i > 10000:
		break
	inputs, labels = data
	print i
	labels = Variable(labels.cuda())
	inputs = Variable(inputs.cuda())
	#print(inputs)
	outputs = encoder(inputs)
	outputs -= 0.5 
	#print outputs
	outputs = F.relu(outputs.sign())
	#print outputs
	binary_test_dict["test" + str(i)] = outputs.data
	testdict["test" + str(i)] = inputs.data
	labeldict["test" + str(i)] = labels.data
"""	
for i,data in enumerate(trainloader):
	print i
	if (i>2):
		break
	inputs1,label1=data
	inputs1 ,label1=Variable(inputs1),Variable(label1)
	
	inputs, labels = data
	inputs,labels = Variable(inputs.cuda()), Variable(labels.cuda())
	outputs = encoder(inputs)
	outputs = F.relu(outputs.sign())
	binary_train_dict["test" + str(i)] = outputs.data
	traindict["test" + str(i)] = inputs1.data
"""


query_key = ["test" + str(randint(0,1000)),"test" + str(randint(0,1000)),"test" + str(randint(0,1000)),"test" + str(randint(0,1000)),"test" + str(randint(0,1000)),"test" + str(randint(0,1000)),"test" + str(randint(0,1000)),"test" + str(randint(0,1000)),"test" + str(randint(0,1000)),"test" + str(randint(0,1000))]

#print(testdict)
for flag in range(10):
	query = testdict[query_key[flag]]
	#print testdict['test9']
	#query=testdict['test9']
	print " Query Image selected "
	#print query.size()

	print "Creating Condition Pool"
	condition_pool ={}
	
	for key, value in testdict.iteritems():
		 
		binary =binary_test_dict[str(key)]
		q_binary = binary_test_dict[query_key[flag]]
		#print binary
		#print q_binary
		hd = binary != q_binary
		#print hd
		hd = hd.sum()
		condition_pool[key] = hd
	#print condition_pool
	sorted_x = sorted(condition_pool.items(), key=operator.itemgetter(1))[0:100] #import operator
	#print sorted_x 
	#print "Saving Image... at /home/deepkliv/Desktop/AE/ram/AE_classifier/fashion/Test_results/Retrieval"
	for i,data in enumerate(sorted_x[1:]):
		if i == 16:
			break
		#print data
		key,value = data
		#print testdict[key][0]
		image = testdict[key][0]
		#print image.size()
		#print query.size()
		if i ==0:
			image_tensor = torch.Tensor(16,3,224,224)
			image_tensor[i] = query
			#print image_tensor.size()
		else:
			image_tensor[i] = image#torch.cat((image_tensor,image),0)
	#print image_tensor.size()
	retreival_path = "/home/siplab/Saket/fashion/Retrieval/result"+str(flag)+ ".png"
	torchvision.utils.save_image(image_tensor, retreival_path)


