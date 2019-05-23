from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import argparse
import numpy as np
# import zipfile
import net_sphere




parser = argparse.ArgumentParser(description='ASC19 face super resolution evaluation code. ')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--HR_dir','-H', default='', type=str)
parser.add_argument('--SR_dir','-S', default='', type=str)
parser.add_argument('--model','-m', default='sphere20a.pth', type=str)
args = parser.parse_args()

predicts=[]
net = getattr(net_sphere,args.net)()
net.load_state_dict(torch.load(args.model))
#net.cuda()
net.eval()
net.feature = True


fls = os.listdir(args.SR_dir)

ave = 0.

for ii, fl in enumerate(fls):
    # print("parsing {}".format(fl))
    name1 = os.path.join(args.HR_dir, fl)
    name2 = os.path.join(args.SR_dir, fl)
    # print(name1, name2)
    img1 = cv2.resize(cv2.imread(name1), (96,112),interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(cv2.imread(name2), (96,112),interpolation=cv2.INTER_CUBIC)
    #print(np.shape(img1))


    imglist = [img1,img2]
    for i in range(len(imglist)):
        imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1,3,112,96))
        imglist[i] = (imglist[i]-127.5)/128.0
    img = np.vstack(imglist)
    with torch.no_grad():
        img = Variable(torch.from_numpy(img).float()) # .cuda()
        #img = Variable(torch.from_numpy(img).float(),volatile=True) # .cuda()
    output = net(img)
    f = output.data
    f1,f2 = f[0],f[1]

    cosdistance = f1.dot(f2)/ f1.norm()/ f2.norm()
    ave = ave + cosdistance
    # print(cosdistance.data, ave/(ii+1))
    # sys.exit()
    # predicts.append('{}\t{}\t{}\t{}\n'.format(name1,name2,cosdistance,sameflag))


print("Final Identity Similarity Score is: {:.4f}".format(ave/len(fls)))
# print("{:.4f}".format(ave/len(fls)))
