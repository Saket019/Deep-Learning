import pickle
import matplotlib
matplotlib.use('agg')
import pylab as plt 
import torch

def dictmod(dict):
    newdict = {}
    count = 1
    newval = 0
    for key,value in dict.items():
        # print(key.split("_")[0])
        if int(key.split("_")[0]) == count:
            newval += value*250
            # print("count",count)
        else: 
            newval = newval/1250
            newdict[count-1] = newval
            newval = 0
            newval += value*250
            count += 1
    newval = newval/1250
    newdict[count-1] = newval
    return newdict
 
# dictdump = pickle.load(open(("/home/saket/CVPRW/log_classifier_mod/test_loss_log.pkl"),"rb"))
# dictdump2 = pickle.load(open(("/home/saket/CVPRW/log_classifier_mod/d1_loss_log.pkl"),"rb"))
# dictdump3 = pickle.load(open(("/home/saket/CVPRW/log_classifier_mod/loss_log.pkl"),"rb"))
# dictdump4 = pickle.load(open(("/home/saket/CVPRW/log_classifier_mod/d2_loss_log.pkl"),"rb"))
# dictdump5 = pickle.load(open(("/home/saket/CVPRW/log_classifier_mod/loss_cls_log.pkl"),"rb"))

# dictdump = pickle.load(open(("/home/saket/CVPRW/log_vanilla_cauchy_olddata/acc_log.pkl"),"rb"))
dictdump = pickle.load(open(("/home/saket/CVPRW/log_vanilla_cauchy_olddata/test_loss_log.pkl"),"rb"))
print(dictdump)
# dictdump3 = pickle.load(open(("/home/saket/CVPRW/log_vanilla_cauchy/cls_acc_log.pkl"),"rb"))



# dictdump = pickle.load(open(("/home/saket/CVPRW/log_classifier_mod/d1_acc_log.pkl"),"rb"))
# dictdump2 = pickle.load(open(("/home/saket/CVPRW/log_classifier_mod/d2_acc_log.pkl"),"rb"))
# dictdump3 = pickle.load(open(("/home/saket/CVPRW/log_classifier_mod/cls_acc_log.pkl"),"rb"))

# print(dictdump2)

xaxis = [0]*len(dictdump.keys())
yaxis = [0]*len(dictdump.keys())
# yaxis2 = [0]*len(dictdump.keys())
# yaxis3 = [0]*len(dictdump.keys())
# yaxis4 = [0]*len(dictdump.keys())
# yaxis5 = [0]*len(dictdump.keys())

for keys,value in dictdump.items():
    # yaxis[keys-1] = value
    # xaxis[keys-1] = keys
    yaxis[keys] = value
    xaxis[keys] = keys
    # yaxis2[keys] = dictmod(dictdump2)[keys]
    # yaxis5[keys] = dictmod(dictdump5)[keys]
    # yaxis3[keys] = dictmod(dictdump3)[keys]
    # yaxis4[keys] = dictmod(dictdump4)[keys]

# plt.figure()
plt.plot(xaxis,yaxis,label= "test")
# plt.plot(xaxis,yaxis2,label='cauchy')
# plt.plot(xaxis,yaxis3, label='cauchy')
# plt.plot(xaxis,yaxis4, label='d2')
# plt.plot(xaxis,yaxis5, label='classifier')

plt.xlabel("Epochs")
plt.ylabel("Loss")
# plt.legend()
plt.savefig("/home/saket/CVPRW/log_vanilla_cauchy_olddata/losstestplot.jpg")
plt.show()
