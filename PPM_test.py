from ScalableTopologicalRegularizers.PPM import *

import sys
import random
import time
import datetime
import pickle

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from gudhi.dtm_rips_complex import DTMRipsComplex

from FiltrationLearningForPointClouds.scripts.lib.pc_representation import *
from FiltrationLearningForPointClouds.scripts.lib.scheduler import *
from expected_toporep import ExpTopoRep
import time

import warnings
warnings.filterwarnings("ignore")



dataset = "ModelNetNoisy01_C=10,N=100,T=1,K=2000"
all_X = torch.load(f"FiltrationLearningForPointClouds/data/{dataset}_data").to(torch.float32)
data_num = all_X.shape[0]
num_points = all_X.shape[1]
all_X = pointcloud_normalize(all_X[:, :num_points, :])
all_y = torch.load(f"FiltrationLearningForPointClouds/data/{dataset}_label")
data_num = all_X.shape[0]

data_idx = list(range(data_num))
data_idx.sort(key=lambda i: (all_y[i], i))
    
CV_idx = 0
train_idx = [data_idx[i] for i in range(data_num) if i % 5 != CV_idx]
trainX = all_X[train_idx, :, :]
trainy = all_y[train_idx]
valid_idx = [data_idx[i] for i in range(data_num) if i % 5 == CV_idx]
validX = all_X[valid_idx, :, :]
validy = all_y[valid_idx]

print(validX.shape)

D = compute_ppm(validX[0,:,:],max_order=1,num_samples=1000)
print(D[0].shape)