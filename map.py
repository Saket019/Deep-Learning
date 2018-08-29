import caffe
import numpy as np
import os
import mnist_reader
from PIL import Image
import pickle
from pylab import *
#from readlmdb import readlmdb
import lmdb
import cv2
from caffe.proto import caffe_pb2


lmdb_env = lmdb.open('/home/siplab/Img/retrieval_val_lmdb')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()

model = '/home/siplab/caffe/models/resnet/resnet50_deploy.prototxt'
weights = '/home/siplab/caffe/resnet_retrieval_50_iter_50000.caffemodel'

model_pred = '/home/siplab/caffe/models/resnet/resnet-50_pred.prototxt'

accuracy_model = ''
caffe.set_mode_gpu()
caffe.set_device(0)

net = caffe.Net(model,weights,caffe.TEST)
net2 = caffe.Net(model_pred,weights,caffe.TEST)
f7 = {}
binarydict = {}
for key, value in lmdb_cursor:
	datum.ParseFromString(value)
	label = datum.label
	data = caffe.io.datum_to_array(datum)
	img = np.transpose(data, (0,1,2))
        img= np.resize(img,(3,224,224))
	labeldict[key] = value
	net.blobs['data'].data[0,:,:,:] = img
	res = net.forward()
	f7data = net.blobs['fc1000_saket'].data
	f7[key] = f7data
	hashcode = np.round(net.blobs['hidden_saket_encode'].data[0])
	binarydict[key] = hashcode
        print key
#-------------------------------------Condition Pool-------------------------------------------#
for key1,value1 in binarydict:
	for key2,value2 in binarydict:
		hd = np.sum(value1 = value2) #hamming distance calculation
		cndpool[key2] = 128 - hd
	
	

net2 = caffe.Net(model,weights,caffe.TEST)
mAP = 0




















