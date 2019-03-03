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
from classifier_imageloader import ImageFolder
import pickle

torch.backends.cudnn.benchmark=True

zsize = 32
batch_size = 2
iterations =  1000
learningRate=0.00001
c1 = 0.01

def shuffler(h1,h2):
    sh = torch.randint(0,2,(batch_size,)) #1 means keeping h1 and 0 means keeping h2
    h1new = torch.matmul(torch.diag(1-sh).float(),h2.cpu())+torch.matmul(torch.diag(sh).float(),h1.cpu())
    h2new = torch.matmul(torch.diag(1-sh).float(),h1.cpu())+torch.matmul(torch.diag(sh).float(),h2.cpu())
    return h1new.cuda(),h2new.cuda()




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
		x_c = F.relu(self.fc2(x))
		x_out = torch.tanh(self.fc3(x_c))
		return x_out,x_c #,indices1,indices2,indices3
    
encoder = Encoder()

class Classifier1(nn.Module):
	def __init__(self):
		super(Classifier1,self).__init__()
		# self.conv1d = nn.Conv1d(2,1,1)
		self.fc1 = nn.Linear(4096, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 8)

	def forward(self,x):
		# x = F.relu(self.conv1d(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.log_softmax(self.fc3(x),dim = 0)
		return x

classifier1 = Classifier1()

class Classifier2(nn.Module):
	def __init__(self):
		super(Classifier2,self).__init__()
		# self.conv1d = nn.Conv1d(2,1,1)
		self.fc1 = nn.Linear(zsize, 128)
		self.fc2 = nn.Linear(128, 256)
		self.fc3 = nn.Linear(256, 128)
		self.fc4 = nn.Linear(128, 3)

	def forward(self,x):
		# x = F.relu(self.conv1d(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.log_softmax(self.fc4(x),dim = 0)
		return x

classifier2 = Classifier2()


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1d = nn.Conv1d(2,1,1)
        self.fc1 = nn.Linear(zsize, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self,x):
        x = F.relu(self.conv1d(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        # print("res",x)
        return x

discriminator = Discriminator()

load_model = False

if load_model:
    print("\n\n Loading Pretrained Model ....... \n\n\n")
    model_path = "/home/saket/Saket/new_models/encoder-129.pkl"
    encoder.load_state_dict(torch.load(model_path))
    disc1.load_state_dict(torch.load(model_path))
    disc2.load_state_dict(torch.load(model_path))
    classifier.load_state_dict(torch.load(model_path))
    loss_dict = pickle.load(open("/home/saket/Saket/new_models/loss_log.pkl","rb"))
    loss_disc_dict = pickle.load(open("/home/saket/Saket/new_models/loss_log.pkl","rb"))
    loss_c1_dict = pickle.load(open("/home/saket/Saket/new_models/loss_log.pkl","rb"))
    loss_c2_dict = pickle.load(open("/home/saket/Saket/new_models/loss_log.pkl","rb"))
    c1_acc_dict = pickle.load(open("/home/saket/Saket/new_models/loss_log.pkl","rb"))
    c2_acc_dict = pickle.load(open("/home/saket/Saket/new_models/loss_log.pkl","rb"))
    disc_acc_dict = pickle.load(open("/home/saket/Saket/new_models/loss_log.pkl","rb"))
    test_loss_dict = pickle.load(open("/home/saket/Saket/new_models/loss_log.pkl","rb"))

else:
    model_path = "/home/saket/Saket/pre-trained_models/alexnet-owt-4df8aa71.pth"
    encoder.load_state_dict(torch.load(model_path), strict = False)
    start = 0
    loss_dict = {}
    loss_disc_dict = {}
    loss_c1_dict = {}
    loss_c2_dict = {}
    test_loss_dict = {}
    disc_acc_dict = {}
    c1_acc_dict = {}
    c2_acc_dict = {}

# print("loss_dict : ",loss_dict)
# print("\n\nStart: ",start)

print(encoder)
print(discriminator)
print(classifier1)
print(classifier2)

print("\nLearning Rate %f  | Batch Size  %d \n\n"%(learningRate,batch_size))

if torch.cuda.is_available():
    encoder.cuda()
    discriminator.cuda()
    classifier1.cuda()
    classifier2.cuda()
    
plt.ion()

transform = transforms.Compose(
	[
	transforms.Resize((224,224), interpolation=2),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	])


trainset = ImageFolder("/home/saket/CVPRW/MVC/Mens/train", transform = transform, target_transform = None)
trainloader = torch.utils.data.DataLoader(trainset, shuffle = True , batch_size = batch_size , num_workers = 4)
testset= ImageFolder("/home/saket/CVPRW/MVC/Mens/test", transform = transform, target_transform = None)
testloader = torch.utils.data.DataLoader(testset, shuffle = True , batch_size = int(batch_size/2) , num_workers = 4)
print("\nDataset generated. \n\n")

# trainset = torchvision.datasets.CIFAR10("/home/saket/Saket/old_data/CIFAR/train",train =True,download = False,transform = transform)    
# trainloader = torch.utils.data.DataLoader(trainset, shuffle = True , batch_size = batch_size , num_workers = 2)
# testset = torchvision.datasets.CIFAR10("/home/saket/Saket/old_data/CIFAR/test", train = False,download = False,transform = transform)
# testloader = torch.utils.data.DataLoader(testset, shuffle = True , batch_size = batch_size, num_workers = 2)

loss_criterion = nn.BCELoss()
c1_loss_criterion = nn.NLLLoss()
c2_loss_criterion = nn.NLLLoss()
disc_criterion = nn.BCEWithLogitsLoss()

# disc1_criterion = nn.BCEWithLogitsLoss()
# disc2_criterion = nn.BCEWithLogitsLoss()

encoder_optimizer = optim.Adam(encoder.parameters(), lr = learningRate,eps = 0.0001,amsgrad = True)
disc_optimizer = optim.Adam(discriminator.parameters(), lr = learningRate,eps = 0.0001,amsgrad = True)
classifier1_optimizer = optim.Adam(classifier1.parameters(), lr = learningRate,eps = 0.0001,amsgrad = True)
classifier2_optimizer = optim.Adam(classifier2.parameters(), lr = learningRate,eps = 0.0001,amsgrad = True)

encoder.train()

for epoch in range(start,iterations):
    run_loss = 0
    run_loss_temp = 0
    run_disc_loss = 0
    run_disc_loss_temp = 0
    run_c1_loss = 0
    run_c1_loss_temp = 0
    run_c2_loss = 0
    run_c2_loss_temp = 0
    c1_run = 0
    c1_run_full = 0
    c2_run =0
    c2_run_full = 0
        
    for i,data in enumerate(trainloader,0):
        # if i > 2:
        #     break
        input1, input2, labels, groundtruths1, groundtruths2 = data

        #########################
        ##### Discriminator #####
        #########################
        """
        indexes = np.where(labels.numpy()==2)[0].tolist()
        if i%2==0 and not len(indexes)==batch_size:
            # f_labels = torch.from_numpy(np.delete(labels.numpy(),indexes,0))
            f_labels  = labels[labels!=2]
            f_input1 = torch.from_numpy(np.delete(input1.numpy(),indexes,0))
            f_input2 = torch.from_numpy(np.delete(input2.numpy(),indexes,0))
            f_input1,f_input2,f_labels = Variable(f_input1).cuda(), Variable(f_input2).cuda(), Variable(f_labels).cuda()
            f_h1 = encoder(f_input1)
            f_h2 = encoder(f_input2)
            del(f_input1)
            del(f_input2)
            #shuffler to be added
            # print(f_h1,f_h2)

            d_input = torch.stack((f_h1,f_h2),1)#.unsqueeze(0)
            d_out = disc1(d_input)
            d_loss = disc1_criterion(torch.squeeze(d_out),f_labels.float())
            run_d1_loss += d_loss.item()
            run_d1_loss_temp += d_loss.item()
            adv_loss = c1*(1-d_loss)

            disc1_optimizer.zero_grad()
            d_loss.backward(retain_graph = True)
            disc1_optimizer.step()

            encoder_optimizer.zero_grad()
            adv_loss.backward()
            encoder_optimizer.step()
            d1_run +=1
            d1_run_full +=1
        """
        #########################
        #########################

        input1,input2, labels = Variable(input1).cuda(), Variable(input2).cuda(), Variable(labels).cuda()
        groundtruths1, groundtruths2 = Variable(groundtruths1).cuda(), Variable(groundtruths2).cuda()
        
        h1,x1 = encoder(input1)
        h2,x2 = encoder(input2)
        
        #########################
        #####  CLASSIFIER1  #####
        #########################
        
        pred = classifier1(x1)
        c1_loss = c1_loss_criterion(pred , groundtruths1)

        classifier1_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        
        c1_loss.backward(retain_graph = True)
        
        classifier1_optimizer.step()
        encoder_optimizer.step()
        
        run_c1_loss += c1_loss.item()
        run_c1_loss_temp += c1_loss.item()

        #########################
        #####  CLASSIFIER2  #####
        #########################
        h1,h2 =  shuffler(h1,h2) 
        c2_input = torch.stack((h1, h2),1) # 0 means shuffled the channel, 1 means not shuffled
        pred = classifier2(c2_input)

        c2_loss = c2_loss_criterion(pred, labels)

        classifier2_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        
        c2_loss.backward(retain_graph = True)
        
        classifier2_optimizer.step()
        encoder_optimizer.step()
        
        run_c2_loss += c2_loss.item()
        run_c2_loss_temp += c2_loss.item()

        #########################
        #########################


        

        del(input1)
        del(input2)

        cos = F.cosine_similarity(h1, h2,dim=1, eps=1e-6)
        dist = (1-cos)*zsize/2
        # print(dist) 

        gamma = 5 #hyperparameter
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
            print('[%d, %d] Cauchy loss: %.3f Discriminator1 loss: %.3f Discriminator2 loss: %.3f classifier loss: %.3f' % (epoch + 1, i + 1 , run_loss_temp/loss_count,run_d1_loss_temp/d1_run,run_d2_loss_temp/d2_run, run_cls_loss_temp/loss_count))
            run_d1_loss_temp = 0
            d1_run = 0
            run_d2_loss_temp = 0
            d2_run = 0
            run_loss_temp = 0.0
            run_cls_loss_temp = 0.0
            key = str((epoch+1))+"_"+str(i)
            loss_dict[key] = run_loss/i
            loss_cls_dict[key] = run_cls_loss/i
            run_loss = 0.0

            loss_d1_dict[key] = run_d1_loss/d1_run_full
            run_d1_loss = 0.0
            
            loss_d2_dict[key] = run_d2_loss/d2_run_full
            run_d2_loss = 0.0
        # break
        #likelihood = classification_criterion(cauchy_output,torch.tensor(s))
        #print("likelihood:", likelihood)

        # print("input1",inputs[0])
        # print("input2",inputs[1])

        # x = Variable(x).cuda()
        # y = Variable(y).cuda()
        # h1 = encoder(x)
        # h2 = encoder(y)

    ###############################

    #SAVING WEIGHTS AND LOSS FILES#

    ###############################

    data_store_path = '/home/saket/CVPRW/log_classifier_mod'
    if not os.path.exists(data_store_path):
        os.makedirs(data_store_path)

    encoder_path = os.path.join(data_store_path, 'encoder-%d.pkl' %(epoch+1))
    torch.save(encoder.state_dict(), encoder_path)
    print("Saving encoder weights to ",encoder_path)
    
    d1_path = os.path.join(data_store_path, 'disc1-%d.pkl' %(epoch+1))
    torch.save(disc1.state_dict(), d1_path)
    print("Saving disc1 weights to ",d1_path)

    d2_path = os.path.join(data_store_path, 'disc2-%d.pkl' %(epoch+1))
    torch.save(disc2.state_dict(), d2_path)
    print("Saving disc2 weights to ",d2_path)

    cls_path = os.path.join(data_store_path, 'classifier-%d.pkl' %(epoch+1))
    torch.save(classifier.state_dict(), cls_path)
    print("Saving classifier weights to ",cls_path)

    loss_log_path = os.path.join(data_store_path, 'loss_log.pkl')
    with open(loss_log_path, 'wb') as handle:
        pickle.dump(loss_dict, handle)
    print("Saving cauchy loss log to ",loss_log_path)

    loss_log_path = os.path.join(data_store_path, 'loss_cls_log.pkl')
    with open(loss_log_path, 'wb') as handle:
        pickle.dump(loss_cls_dict, handle)
    print("Saving classiier loss log to ",loss_log_path)
    
    loss_log_path = os.path.join(data_store_path, 'd1_loss_log.pkl')
    with open(loss_log_path, 'wb') as handle:
        pickle.dump(loss_d1_dict, handle)
    print("Saving Disc1 loss log to ",loss_log_path)
    
    loss_log_path = os.path.join(data_store_path, 'd2_loss_log.pkl')
    with open(loss_log_path, 'wb') as handle:
        pickle.dump(loss_d2_dict, handle)
    print("Saving Disc2 loss log to ",loss_log_path)
    ################################

    ###          Testing         ###

    ################################
    
    if ( epoch+1 )% 1 == 0:
        with torch.no_grad():
            total = 0
            t_loss = 0
            correct = 0
            d1_total = 0
            d1_correct = 0
            d2_total = 0
            d2_correct = 0
            
            print('\n Testing ....')
            for t_i,t_data in enumerate(testloader):
                if t_i > 100:
                    break
                
                t_input1 ,t_input2 ,t_labels ,t_gt1 ,t_gt2 = t_data

                indexes = np.where(t_labels.numpy()==2)[0].tolist()
                #d1 accuracy
                if not len(indexes)==batch_size:
                    ft_input1 = torch.from_numpy(np.delete(t_input1.numpy(),indexes,0)) 
                    ft_input2 = torch.from_numpy(np.delete(t_input2.numpy(),indexes,0))
                    d1_labels  = t_labels[t_labels!=2]
                    ft_input1,ft_input2,d1_labels = Variable(ft_input1).cuda(), Variable(ft_input2).cuda(), Variable(d1_labels).cuda()                    
                    h1_ft = encoder(ft_input1)
                    h2_ft = encoder(ft_input2)
                    d_input = torch.stack((h1_ft,h2_ft),1)#.unsqueeze(0)
                    d1_out = torch.squeeze(disc1(d_input))
                    d1_new_out = d1_out>0.5
                    # print(d1_out)
                    # print(d1_labels)
                    d1_total += len(d1_labels)
                    d1_correct += len(d1_labels) - torch.sum(d1_new_out^d1_labels.byte())



                t_input1,t_input2, t_labels = Variable(t_input1).cuda(), Variable(t_input2).cuda(), Variable(t_labels).cuda()
                t_gt1,t_gt2 = Variable(t_gt1).cuda(), Variable(t_gt2).cuda()
                s_t = (t_labels==2)
                h1_t = torch.tanh(encoder(t_input1))
                h2_t = torch.tanh(encoder(t_input2))
                #classification accuracy
                t_pred = classifier(h1_t)
                # print("t_pred",t_pred)
                _, predicted = torch.max(t_pred.data , 1)
                # print("predicted",predicted)
                total += len(t_gt1)
                correct += (predicted == t_gt1).sum().cpu().numpy()

                #d2 accuracy
                d2_input = torch.stack((h1_t,h2_t),1)
                d2_out = torch.squeeze(disc2(d2_input))
                d2_new_out =  d2_out > 0.5
                d2_labels  = (t_labels==2)
                d2_total += len(d2_labels)
                d2_correct += len(d2_labels) - torch.sum(d2_new_out ^ d2_labels)

                cos = F.cosine_similarity(h1_t, h2_t, dim=1, eps=1e-6)
                dist = (1-cos)*zsize/2
                # print(dist) 

                gamma = 5 #hyperparameter
                cauchy_output = torch.reciprocal(dist+gamma)*gamma

                try:
                    t_loss += loss_criterion(torch.squeeze(cauchy_output),s_t.float()).item()
                except RuntimeError:
                    print("s",torch.max(s_t.float()).item(),torch.min(s_t.float()).item())
                    print("\nCO ",torch.max(torch.squeeze(cauchy_output)).item(),torch.min(torch.squeeze(cauchy_output)).item())
                del(t_input1)
                del(t_input2)

            print('Classification Accuracy of the network on the test images: %.5f \n\n' % ((100.0*correct) / total))
            print('D1 Classification Accuracy of the network on the test images: %.5f \n\n' % ((100.0*d1_correct) / d1_total))
            print('D2 Classification Accuracy of the network on the test images: %.5f \n\n' % ((100.0*d2_correct) / d2_total))
            
            test_loss_dict[epoch] = t_loss/t_i
            print("\nTesting loss: %.5f" % (t_loss/t_i))
            test_loss_log_path = os.path.join(data_store_path, 'test_loss_log.pkl')
            with open(test_loss_log_path, 'wb') as handle:
                pickle.dump(test_loss_dict, handle)

            cls_acc_dict[epoch] = (100.0*correct) / total
            classifier_accuracy_log_path = os.path.join(data_store_path, 'cls_acc_log.pkl')
            with open(classifier_accuracy_log_path, 'wb') as handle:
                pickle.dump(cls_acc_dict, handle)

            d1_acc_dict[epoch] = (100.0*d1_correct) / d1_total
            d1_accuracy_log_path = os.path.join(data_store_path, 'd1_acc_log.pkl')
            with open(d1_accuracy_log_path, 'wb') as handle:
                pickle.dump(d1_acc_dict, handle)

            d2_acc_dict[epoch] = (100.0*d2_correct) / d2_total
            d2_accuracy_log_path = os.path.join(data_store_path, 'd2_acc_log.pkl')
            with open(d2_accuracy_log_path, 'wb') as handle:
                pickle.dump(d2_acc_dict, handle)

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