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
from imageloader import ImageFolder

import pickle

torch.backends.cudnn.benchmark=True
zsize = 32
batch_size = 256
iterations =  1000
learningRate=0.00001
gamma = 5 #hyperparameter


def cauchy(x, gamma = torch.ones(1).cuda()):
    return Variable(gamma/(gamma+x), requires_grad = True)

def hd(h1,h2):
    h1norm = Variable(torch.div(h1,torch.norm(h1,p=2)),requires_grad = True)
    h2norm = Variable(torch.div(h2,torch.norm(h2,p=2)),requires_grad = True)
    return Variable(torch.pow(torch.norm(h1norm - h2norm, p=2),2),requires_grad = True)

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
		# print(x.shape)
		x = F.relu(self.conv1(x))
		x, indices1 = F.max_pool2d(x,(3,3),(2,2),return_indices = True)
		x = F.relu(self.conv2(x))
		x, indices2 = F.max_pool2d(x,(3,3),(2,2),return_indices = True)
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = F.relu(self.conv5(x))
		x, indices3 = F.max_pool2d(x,(3,3),(2,2),return_indices = True)
		x = x.view(x.size(0), 256 * 6 * 6)
		x = F.dropout(x)
		x = F.relu(self.fc1(x))
		x = F.dropout(x)
		x = F.relu(self.fc2(x))
		x = torch.tanh(self.fc3(x))
		return x #,indices1,indices2,indices3
    
encoder = Encoder()

load_model = False

if load_model:
    print("\n\n Loading Pretrained Model ....... \n\n\n")
    model_path = "/home/saket/Saket/new_models/encoder-129.pkl"
    encoder.load_state_dict(torch.load(model_path))
    start = int(model_path.split("/")[-1].split('.')[0].split('-')[-1])
    loss_dict = pickle.load(open(("/home/saket/Saket/new_models/loss_log-%d.pkl" % start),"rb"))

else:
    model_path = "/home/saket/Saket/pre-trained_models/alexnet-owt-4df8aa71.pth"
    encoder.load_state_dict(torch.load(model_path), strict = False)
    start = 0
    loss_dict = {}
    loss_cls_dict = {}
    test_loss_dict = {}
    acc_dict = {}

print(encoder)

if torch.cuda.is_available():
    encoder.cuda()

print("\nLearning Rate %f  | Batch Size  %d \n\n"%(learningRate,batch_size))

plt.ion()

transform = transforms.Compose(
	[
	transforms.Resize((224,224), interpolation=2),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	])


trainset = ImageFolder("/home/saket/Saket/old_data/train", transform = transform, target_transform = None)
# print(trainset)
# trainset=torchvision.datasets.ImageFolder("/home/saket/CVPRW/MVC/Mens/train", transform = transform, target_transform = None)
trainloader = torch.utils.data.DataLoader(trainset, shuffle = True , batch_size = batch_size , num_workers = 4)
testset= ImageFolder("/home/saket/Saket/old_data/test", transform = transform, target_transform = None)
testloader = torch.utils.data.DataLoader(testset, shuffle = True , batch_size = int(batch_size/2) , num_workers = 4)
print("\nDataset generated. \n\n")

# trainset = torchvision.datasets.CIFAR10("/home/saket/Saket/old_data/CIFAR/train",train =True,download = False,transform = transform)    
# trainloader = torch.utils.data.DataLoader(trainset, shuffle = True , batch_size = batch_size , num_workers = 2)
# testset = torchvision.datasets.CIFAR10("/home/saket/Saket/old_data/CIFAR/test", train = False,download = False,transform = transform)
# testloader = torch.utils.data.DataLoader(testset, shuffle = True , batch_size = batch_size, num_workers = 2)

loss_criterion = nn.BCELoss()

encoder_optimizer = optim.Adam(encoder.parameters(), lr = learningRate, eps = 0.0001, amsgrad = True)


encoder.train()

for epoch in range(start,iterations):
    run_loss = 0
    run_loss_temp = 0
    run_cls_loss = 0
    run_cls_loss_temp = 0
    run_d1_loss = 0
    run_d1_loss_temp = 0
    run_d2_loss = 0
    run_d2_loss_temp = 0
    d1_run = 0
    d1_run_full = 0
    d2_run =0
    d2_run_full = 0
        
    for i,data in enumerate(trainloader,0):
        # if i > 2:
        #     break
        input1,input2, labels = data
        input1,input2, labels = Variable(input1).cuda(), Variable(input2).cuda(), Variable(labels).cuda()

        s = (labels==2)

        h1 = encoder(input1)
        h2 = encoder(input2)

        del(input1)
        del(input2)

        cos = F.cosine_similarity(h1, h2,dim=1, eps=1e-6)
        dist = (1-cos)*zsize/2
        
        cauchy_output = torch.reciprocal(dist+gamma)*gamma

        try:
            loss = loss_criterion(torch.squeeze(cauchy_output),s.float())
        except RuntimeError: 
            print("s",torch.max(s.float()).item(),torch.min(s.float()).item())
            print("\nCO ",torch.max(torch.squeeze(cauchy_output)).item(),torch.min(torch.squeeze(cauchy_output)).item())

        encoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        run_loss += loss.item()
        run_loss_temp += loss.item()
        # print("\n\n\n gradient started...\n\n")
        # for name,param in encoder.named_parameters():
        #     print(name))
        #     print("Grad Output",torch.max(param.grad),torch.min(param.grad))
        # print("\n\n\n gradient ended...\n\n")

        loss_count = 250 #after how many iterations do you want to show loss
        if (i +1) % loss_count == 0:
            print('[%d, %d] Cauchy loss: %.3f ' % (epoch + 1, i + 1 , run_loss_temp/loss_count))
            run_loss_temp = 0.0
            key = str((epoch+1))+"_"+str(i)
            loss_dict[key] = run_loss/i
            run_loss = 0.0

    data_store_path = '/home/saket/CVPRW/log_vanilla_cauchy_olddata'
    if not os.path.exists(data_store_path):
        os.makedirs(data_store_path)

    encoder_path = os.path.join(data_store_path, 'encoder-%d.pkl' %(epoch+1))
    torch.save(encoder.state_dict(), encoder_path)
    print("Saving encoder weights to ",encoder_path)
    
    loss_log_path = os.path.join(data_store_path, 'loss_log.pkl')
    with open(loss_log_path, 'wb') as handle:
        pickle.dump(loss_dict, handle)
    print("Saving cauchy loss log to ",loss_log_path)
    
    if ( epoch+1 )% 1 == 0:
        with torch.no_grad():
            total = 0
            t_loss = 0
            correct = 0
            
            print('\n Testing ....')
            for t_i,t_data in enumerate(testloader):
                if t_i > 100:
                    break
                
                t_input1, t_input2, t_labels = t_data

                t_input1,t_input2, t_labels = Variable(t_input1).cuda(), Variable(t_input2).cuda(), Variable(t_labels).cuda()
                s_t = (t_labels==2)
                h1_t = torch.tanh(encoder(t_input1))
                h2_t = torch.tanh(encoder(t_input2))

                cos = F.cosine_similarity(h1_t, h2_t, dim=1, eps=1e-6)
                dist = (1-cos)*zsize/2
                #print(dist)

                gamma = 4 #hyperparameter
                cauchy_output = torch.reciprocal(dist+gamma)*gamma
                # print(cauchy_output)
                new_cauchy_output = cauchy_output>0.5
                
                total += len(new_cauchy_output) 
                # print(new_cauchy_output^s_t)
                correct += len(new_cauchy_output)- (new_cauchy_output ^ s_t).sum()
                
                try:
                    t_loss += loss_criterion(torch.squeeze(cauchy_output),s_t.float()).item()
                except RuntimeError:
                    print("s ",torch.max(s_t.float()).item(),torch.min(s_t.float()).item())
                    print("\nCO ",torch.max(torch.squeeze(cauchy_output)).item(),torch.min(torch.squeeze(cauchy_output)).item())

                del(t_input1)
                del(t_input2)

            print('Accuracy of the network on the test images: %.5f \n\n' % ((100.0*correct) / total))
            
            test_loss_dict[epoch] = t_loss/t_i
            print("\nTesting loss: %.5f" % (t_loss/t_i))
            test_loss_log_path = os.path.join(data_store_path, 'test_loss_log.pkl')
            with open(test_loss_log_path, 'wb') as handle:
                pickle.dump(test_loss_dict, handle)

            acc_dict[epoch] = (100.0*correct) / total
            acc_log_path = os.path.join(data_store_path, 'acc_log.pkl')
            with open(acc_log_path, 'wb') as handle:
                pickle.dump(acc_dict, handle)
"""
    encoder_optimizer.zero_grad()		
    class_pred = classification(inputs)
    print("class_pred",class_pred)
    print("class_pred_size",class_pred.size())
    print("labels",labels)
    print(labels.size())
    c_loss = classification_criterion(class_pred , labels)
    print("c_loss",c_loss)
    c_loss.backward(retain_graph=True)
    classification_optimizer.step()
    
    run_c_loss += c_loss.data[0]
    #print i
    if (i +1) % 2 == 0:
        print('[%d, %5d] Classification loss: %.3f' % (epoch + 1, i + 1 , run_c_loss/2))
        run_c_loss = 0.0
"""