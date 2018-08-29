from matplotlib import pyplot as plt 
from PIL import Image
import caffe
import lmdb
import numpy as np
import cv2
from caffe.proto import caffe_pb2
import pickle
import operator

m = 1000 #no. of coarse level searches
k = 100 #no. of fine level searches

lmdb_env = lmdb.open('/home/siplab/dress_length/dresslength_test_lmdb')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
print lmdb_cursor
datum = caffe_pb2.Datum()
traindata = {}
for key, value in lmdb_cursor:
	datum.ParseFromString(value)
	label = datum.label
	data = caffe.io.datum_to_array(datum)
	image = np.transpose(data, (1,2,0))
	traindata[key] = image

binarycodes = pickle.load(open('/home/siplab/caffe/models/resnet/binary_train.pickle','rb'))
threshold = 100
qimg = "00028064_mini_4174.jpg"

cndpool = {}
for key,value in traindata.items():
	hd = np.sum(binarycodes[qimg] == binarycodes[key]) #hamming distance calculation
	cndpool[key] = 128 - hd
sorted_x = sorted(cndpool.items(), key=operator.itemgetter(1))[0:m]

f7data = pickle.load(open('/home/siplab/caffe/models/resnet/f7_train.pickle','rb'))


dist = {}
for j in range(len(sorted_x)):
	dist[sorted_x[j][0]] = np.linalg.norm( f7data[qimg] - f7data[sorted_x[j][0]] )
results = sorted(dist.items(), key=operator.itemgetter(1))[0:k]

for flag in range(11):
	cv2.imshow(results[flag][0],traindata[results[flag][0]])
	cv2.waitKey(10000)

