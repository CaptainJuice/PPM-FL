import torch
import torch.nn as nn

from FiltrationLearningForPointClouds.scripts.lib.utils import *
from FiltrationLearningForPointClouds.scripts.lib.pointnet import *
from FiltrationLearningForPointClouds.scripts.lib.pointMLP import *
from FiltrationLearningForPointClouds.scripts.lib.deepsets import *
from FiltrationLearningForPointClouds.scripts.lib.toporep import *

class PCFeatureNet(nn.Module):
    def __init__(self, input_dim, num_points, pointnet=False, deepsets=False, pointmlp=False, matds=False, matds_dist=False, ph=False, **kwargs):
        super(PCFeatureNet, self).__init__()


        self.nb_repeat = 20

        self.input_dim = input_dim
        self.num_points = num_points
        self.feature_dim = 16 * (pointnet + deepsets + matds + ph)

        self.deep_fully_connected = kwargs.get("deep_fully_connected", True)

        if pointnet:
            self.pointnet = PointNet(self.input_dim, self.num_points, 16)
        else:
            self.pointnet = None
        if deepsets:
            self.deepsets = DeepSets(self.input_dim, self.num_points, 16)
        else:
            self.deepsets = None
        if pointmlp:
            self.pointmlp = PointMLP(points=self.num_points, class_num=16, input_dim=self.input_dim)
        else:
            self.pointmlp = None
        if matds:
            self.matds = DistNet(1, self.num_points, 16)
        else:
            self.matds = None
        if matds_dist:
            assert self.input_dim is None
            self.matds_dist = MatrixDeepSets(1, num_points, 16, doub=True)
        else:
            self.matds_dist = None
        if ph:
            print("Use Perslay!")
            self.perslay = PersLay(16, num_obsrv=32)
        else:
            self.perslay = None
    
    def forward(self, X, pd_points_list=None, out_vec=None):
        rep_vec_list = [] if out_vec is None else [out_vec]
        if self.deepsets is not None:
            rep_vec_list.append(self.deepsets(X))
        if self.pointnet is not None:
            rep_vec_list.append(self.pointnet(X))
        if self.pointmlp is not None:
            rep_vec_list.append(self.pointmlp(X))
        if self.matds is not None:
            rep_vec_list.append(self.matds(X))
        if self.matds_dist is not None:
            rep_vec_list.append(self.matds_dist(X.unsqueeze(dim=3)))
        if self.perslay is not None:
            # for i in range(self.nb_repeat):
            #     if i == 0:
            #         res = self.perslay([x[i] for x in pd_points_list])
            #         # print('perslay', res.shape)
            #     else:
            #         res = res + self.perslay([x[i] for x in pd_points_list])
            #     res = res / self.nb_repeat
            rep_vec_list.append(self.perslay(pd_points_list))
            # rep_vec_list.append(res)
        return torch.cat(rep_vec_list, dim=1)
    

class DistNet(nn.Module):
    def __init__(self, input_dim, num_points, num_labels, **kwargs):
        super(DistNet, self).__init__()
        self.matds = MatrixDeepSets(input_dim, num_points, num_labels, doub=True, **kwargs)
    
    def forward(self, input_data):
        dist = torch.cdist(input_data, input_data).unsqueeze(dim=3)
        return self.matds(dist)