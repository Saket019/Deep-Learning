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
torch.backends.cudnn.benchmark=True
zsize = 48
batch_size = 259
iterations =  500
learningRate=0.0001

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
		
		return x,indices1,indices2,indices3
encoder = Encoder()

loaded_weights = torch.load('/home/siplab/Saket/alexnet-owt-4df8aa71.pth')

encoder.conv1.weight = loaded_weights['features.0.weight']
encoder.conv1.bias = loaded_weights['features.0.bias']

encoder.conv2.weight = loaded_weights['features.3.weight']
encoder.conv2.bias = loaded_weights['features.3.bias']

encoder.conv3.weight = loaded_weights['features.6.weight']
encoder.conv3.bias = loaded_weights['features.6.bias']

encoder.conv4.weight = loaded_weights['features.8.weight']
encoder.conv4.bias = loaded_weights['features.8.bias']

encoder.conv5.weight = loaded_weights['features.10.weight']
encoder.conv5.bias = loaded_weights['features.10.bias']

encoder.fc1.weight = loaded_weights['classifier.1.weight']
encoder.fc1.bias = loaded_weights['classifier.1.bias']

encoder.fc2.weight = loaded_weights['classifier.4.weight']
encoder.fc2.bias = loaded_weights['classifier.4.bias']


class Binary(Function):

    @staticmethod
    def forward(ctx, input):
        return F.relu(Variable(input.sign())).data

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

binary = Binary()


class Classifier(nn.Module):
	def __init__(self):
		super(Classifier,self).__init__()
		self.L1 = nn.Linear(zsize,128)
		self.L2 = nn.Linear(128,64)
		self.L3 = nn.Linear(64,23)

	def forward(self,x):
		x = F.relu(self.L1(x))
		x = F.relu(self.L2(x))
		x = F.log_softmax(self.L3(x))
		return x


print Classifier()
classifier = Classifier()

class Classification(nn.Module):
	def __init__(self):
		super(Classification,self).__init__()
		self.encoder = Encoder()
		#self.binary = Binary()
		self.classifier = Classifier()

	def forward(self,x):
		x,_,_,_= self.encoder(x)
		#x = binary.apply(x)
		#x,_,_ = self.binary(x)
		x = self.classifier(x)
		return x

print Classification()
classification = Classification()


if torch.cuda.is_available():
	classification.cuda()
	encoder.cuda()
	classifier.cuda()
	#binary.cuda()
#data

plt.ion()

transform = transforms.Compose(
	[
	transforms.Scale((224,224), interpolation=2),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	#transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
	#transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
	])


trainset=torchvision.datasets.ImageFolder("/home/siplab/Saket/fashion/Img/dataset/train", transform=transform, target_transform=None)
trainloader = torch.utils.data.DataLoader(trainset, shuffle = True , batch_size = batch_size , num_workers = 2)
testset=torchvision.datasets.ImageFolder("/home/siplab/Saket/fashion/Img/dataset/test", transform=transform, target_transform=None)
testloader = torch.utils.data.DataLoader(testset, shuffle = True , batch_size = batch_size , num_workers = 2)

classification_criterion = nn.NLLLoss()

classification_optimizer = optim.Adam(classification.parameters(), lr = learningRate)

list_c_loss = []

#fig = plt.figure()
for epoch in range(iterations):
	run_loss = 0 
	run_c_loss = 0

	for i,data in enumerate(trainloader):
		#print i
		inputs, labels = data
		inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

		classification_optimizer.zero_grad()		
		class_pred = classification(inputs)
		c_loss = classification_criterion(class_pred , labels)
		c_loss.backward(retain_graph=True)
		classification_optimizer.step()
		
		run_c_loss += c_loss.data[0]
		#print i
		if (i +1) % 2 == 0:
			print('[%d, %5d] Classification loss: %.3f' % (epoch + 1, i + 1 , run_c_loss/2))
			run_c_loss = 0.0



	encoder_path = os.path.join('/home/siplab/Saket/fashion/Encoder/', 'encoder-%d.pkl' %(epoch+1))
	classifier_path = os.path.join('/home/siplab/Saket/fashion/Classifier/', 'classifier-%d.pkl' %(epoch+1))
	classification_path = os.path.join('/home/siplab/Saket/fashion/','classification-%d.pkl' %(epoch+1))
	
	torch.save(encoder.state_dict(), encoder_path)
	torch.save(classifier.state_dict(), classifier_path)
	torch.save(classification.state_dict(), classification_path)
	
	if ( epoch+1 )% 1 == 0:
		list_c_loss.append(run_c_loss/5000)
		correct = 0
		total = 0
		print('\n Testing ....')
		for t_i,t_data in enumerate(testloader):
			if t_i * batch_size > 2590:
				break
			t_inputs,t_labels = t_data
			t_inputs = Variable(t_inputs).cuda()
			t_labels = t_labels.cuda()
			c_pred = classification(t_inputs)
			_, predicted = torch.max(c_pred.data, 1)
			#print predicted.type() , t_labels.type()
			total += t_labels.size(0)
			correct += (predicted == t_labels).sum()
		print('Accuracy of the network on the 8000 test images: %d %%' % (100 * correct / total))

print('Finished Training and Testing')
