import numpy as np
from gudhi.weighted_rips_complex import WeightedRipsComplex
import torch
import torch.nn as nn
from utils2 import *

from FiltrationLearningForPointClouds.scripts.lib.utils import *
from FiltrationLearningForPointClouds.scripts.lib.pointnet import *
from FiltrationLearningForPointClouds.scripts.lib.deepsets import *

from ScalableTopologicalRegularizers.PPM import *



class ExpTopoRep(nn.Module):
    def __init__(self, num_points, num_labels, **kwargs):
        super(ExpTopoRep, self).__init__()
        
        ### necessary parameters ###
        self.num_points = num_points
        self.num_labels = num_labels

        ### optional parameters ###
        self.method = kwargs.get("method", "dist")
        self.input_format = kwargs.get("input_format", "dist")
        self.reducer = kwargs.get("reducer", False)
        self.perslay = kwargs.get("perslay", not self.reducer)
        # new parameters for expected topological representation
        self.subset_size = kwargs.get("subset_size", 4)
        self.nb_repeat = kwargs.get("nb_repeat",300)
        print("Number of Repeatation = ",self.nb_repeat)
        self.device = kwargs.get("device", "cuda:0")
        
        ### Networks ###
        if self.method == "dist":
            print('Use dist2weight')
            self.weight_net = Dist2Weight(self.num_points, self.subset_size, **kwargs)
        elif self.method == "dist_transformer":
            self.weight_net = Dist2WeightTransformer(self.num_points, **kwargs)
        else:
            raise NotImplementedError
        
        # put the networks on the same device
        self.weight_net.to(self.device)

        if self.reducer:
            self.reducer = nn.Sequential( 
                nn.Linear(self.PI_grid_num ** 2, self.num_labels), 
                nn.BatchNorm1d(self.num_labels), 
            )
        else:
            self.reducer = nn.Identity()
        # if self.perslay:
        #     self.perslay = PersLay(num_labels=self.num_labels, **kwargs)
        # else:
        #     self.perslay = None
    
    # def _get_pd_points(self, dist: torch.Tensor, weight: torch.Tensor):
    #     filt = WeightedRipsComplex(dist.detach().numpy(), weight)
    #     simplex_tree = filt.create_simplex_tree(max_dimension=2)
    #     simplex_tree.compute_persistence()
    #     pers_pair = simplex_tree.persistence_pairs()

    #     _dist = dist.clone()
    #     _dist += torch.stack([weight for _ in range(_dist.shape[0])], dim=0)
    #     _dist += torch.stack([weight for _ in range(_dist.shape[0])], dim=1)
        
    #     points = []
    #     for x in pers_pair:
    #         if len(x[0]) == 2:
    #             birth = _dist[x[0][0], x[0][1]]
    #             death = _dist[x[1][0], x[1][1]]
    #             death = torch.max(death, _dist[x[1][2], x[1][0]])
    #             death = torch.max(death, _dist[x[1][1], x[1][2]])
    #             points.append((birth, death - birth))
    #         elif len(x[0]) >= 3:
    #             raise NotImplementedError
        
    #     return points
    
    # def get_pd_points_list(self, X: torch.Tensor):
    #     """
    #     X: torch.Tensor (num_data x num_points x point_dim)
    #     """
    #     if self.input_format == "dist":
    #         dist = X.clone()
    #     else:
    #         dist = torch.cdist(X, X)

    #     weight = self.weight_net(dist, dist)[:, :, 0]
    #     points_list = [self._get_pd_points(dist[i, :, :], weight[i, :]) for i in range(dist.shape[0])]

    #     return points_list
    
    # def get_pd_points_list_sub(self, X: torch.Tensor):
    #     """
    #     X: torch.Tensor (num_data x num_points x point_dim)
    #     """
    #     if self.input_format == "dist":
    #         dist = X.clone()
    #     else:
    #         dist = torch.cdist(X, X)

    #     # generate distance metrices corresponding to random subsets of point cloud 
    #     dist_new,dist_new_vec,lst_indices = sample_distance_matrices(dist, self.nb_repeat, self.subset_size)
    #     assert len(dist_new.shape) == 4,'Input error, no random distance !!'
    #     assert dist_new.shape[2] == dist_new.shape[3],'Input error, distance shape error !!' 

    #     nb_repeat = dist_new.shape[1]
    #     weight = torch.zeros((dist.shape[0], dist.shape[1]),
    #                          dtype=dist.dtype, device=dist.device)
        
    #     for i in range(nb_repeat):
    #         weight_tmp = self.weight_net(dist_new[:,i,:,:], dist_new_vec[:,i,:,:])
    #         weight = weight + weight_tmp

    #     weight = weight / nb_repeat
    #     # print('weight shape : ' + str(weight.shape))
    #     # assert self.perslay is not None, 'PersLay must not be None !!'

    #     # assign corresponding weights to each point in the subsampled point cloud
    #     sub_weight = torch.zeros((weight.shape[0], weight.shape[1],self.subset_size),
    #                              dtype=weight.dtype, device=weight.device)
        
    #     assert len(lst_indices) == weight.shape[0], 'lst_indices error 1 !!'
    #     assert len(lst_indices[0]) == nb_repeat, 'lst_indices error 2 !!'

    #     for i in range(weight.shape[0]):
    #         for j in range(nb_repeat):
    #             sub_weight[i,j,:] = weight[i,lst_indices[i][j]]
        
    #     # points_list = [sum([self._get_pd_points(dist_new[i,j,:,:], sub_weight[i,j,:]) \
    #     #                     for j in range(nb_repeat)],[]) for i in range(dist.shape[0])]
        
    #     points_list = [[self._get_pd_points(dist_new[i,j,:,:], sub_weight[i,j,:]) \
    #                         for j in range(nb_repeat)] for i in range(dist.shape[0])]
        
    #     return points_list
    
    def get_PPM_rips(self, X: torch.Tensor,Vec: torch.Tensor, Assign_ids):
        """
        X: torch.Tensor (num_data x num_points x point_dim)
        """
        if self.input_format == "dist":
            # dist = X.clone()
            dist = X
            # print('Use dist')
        else:
            dist = torch.cdist(X, X)
            assert False,'input is not dist'
        
        self.nb_repeat = dist.shape[1]
        
        # generate distance metrices corresponding to random subsets of point cloud 
        dist_new = dist
        # dist_new_vec = Vec.clone()
        # dist_new,dist_new_vec,lst_indices = sample_distance_matrices(dist, self.nb_repeat, self.subset_size)
        assert len(dist_new.shape) == 4,'Input error, no random distance !!'
        assert dist_new.shape[2] == dist_new.shape[3],'Input error, distance shape error !!' 

        nb_repeat = dist_new.shape[1]
        # weight = torch.zeros((dist.shape[0], dist_new_vec.shape[2]),
        #                      dtype=dist.dtype, device=dist.device)

        # # print('Build weight success')
        
        # for i in range(nb_repeat):
        #     weight_tmp = self.weight_net(dist_new[:,i,:,:], dist_new_vec[:,i,:,:])
        #     weight = weight + weight_tmp

        # # print('Compute all weights')
        # weight = weight / nb_repeat

        # # assign corresponding weights to each point in the subsampled point cloud
        # sub_weight = torch.zeros((weight.shape[0], nb_repeat,self.subset_size),
        #                          dtype=weight.dtype, device=weight.device)
        
        # assert len(Assign_ids) == weight.shape[0], 'lst_indices error 1 !!'
        # assert len(Assign_ids[0]) == nb_repeat, 'lst_indices error 2 !!'

        # for i in range(weight.shape[0]):
        #     for j in range(nb_repeat):
        #         sub_weight[i,j,:] = weight[i,Assign_ids[i][j]]


        # sub_weight_ = sub_weight.unsqueeze(-2) + sub_weight.unsqueeze(-1)

        # print(sub_weight_[0])
        # print(torch.max(sub_weight_),torch.min(sub_weight_))
            
        # dist_new = dist_new + sub_weight_

        diag_mask = torch.eye(dist_new.shape[-1], dtype=torch.bool).unsqueeze(0).unsqueeze(0).expand(dist_new.shape[0], nb_repeat, -1, -1)

        
        dist_new[diag_mask] = 0

        # print(dist_new)

        # points_lst_2 = [compute_simple_homology(dist_new[i,:,:,:]) for i in range(dist_new.shape[0])]
        # points_lst_2 = []
        # points_lst_1 = [compute_simple_homology(dist_new_off[i,:,:4,:4]) for i in range(dist_new.shape[0])]
        points_lst_1 = [compute_simple_homology(dist_new[i,:,:,:]) for i in range(dist_new.shape[0])]
        # points_lst_1 = []

        points_lst_0 = [compute_simple_homology(dist_new[i,:,:2,:2]) for i in range(dist_new.shape[0])]

        # return points_lst_0,-1

        return points_lst_0,points_lst_1
    
    def get_PPM(self, X: torch.Tensor,Vec: torch.Tensor, Assign_ids):
        """
        X: torch.Tensor (num_data x num_points x point_dim)
        """
        if self.input_format == "dist":
            # dist = X.clone()
            dist = X
            # print('Use dist')
        else:
            dist = torch.cdist(X, X)
            assert False,'input is not dist'
        
        self.nb_repeat = dist.shape[1]
        
        # generate distance metrices corresponding to random subsets of point cloud 
        dist_new = dist
        dist_new_vec = Vec.clone()
        # dist_new,dist_new_vec,lst_indices = sample_distance_matrices(dist, self.nb_repeat, self.subset_size)
        assert len(dist_new.shape) == 4,'Input error, no random distance !!'
        assert dist_new.shape[2] == dist_new.shape[3],'Input error, distance shape error !!' 

        nb_repeat = dist_new.shape[1]
        weight = torch.zeros((dist.shape[0], dist_new_vec.shape[2]),
                             dtype=dist.dtype, device=dist.device)

        # print('Build weight success')
        
        for i in range(nb_repeat):
            weight_tmp = self.weight_net(dist_new[:,i,:,:], dist_new_vec[:,i,:,:])
            weight = weight + weight_tmp

        # print('Compute all weights')
        weight = weight / nb_repeat

        # assign corresponding weights to each point in the subsampled point cloud
        sub_weight = torch.zeros((weight.shape[0], nb_repeat,self.subset_size),
                                 dtype=weight.dtype, device=weight.device)
        
        assert len(Assign_ids) == weight.shape[0], 'lst_indices error 1 !!'
        assert len(Assign_ids[0]) == nb_repeat, 'lst_indices error 2 !!'

        for i in range(weight.shape[0]):
            for j in range(nb_repeat):
                sub_weight[i,j,:] = weight[i,Assign_ids[i][j]]


        sub_weight_ = sub_weight.unsqueeze(-2) + sub_weight.unsqueeze(-1)

        # print(sub_weight_[0])
        # print(torch.max(sub_weight_),torch.min(sub_weight_))
            
        dist_new = dist_new + sub_weight_

        diag_mask = torch.eye(dist_new.shape[-1], dtype=torch.bool).unsqueeze(0).unsqueeze(0).expand(dist_new.shape[0], nb_repeat, -1, -1)

        
        dist_new[diag_mask] = 0

        # print(dist_new)

        # points_lst_2 = [compute_simple_homology(dist_new[i,:,:,:]) for i in range(dist_new.shape[0])]
        # points_lst_2 = []
        # points_lst_1 = [compute_simple_homology(dist_new_off[i,:,:4,:4]) for i in range(dist_new.shape[0])]
        points_lst_1 = [compute_simple_homology(dist_new[i,:,:,:]) for i in range(dist_new.shape[0])]
        # points_lst_1 = []

        points_lst_0 = [compute_simple_homology(dist_new[i,:,:2,:2]) for i in range(dist_new.shape[0])]

        # return points_lst_0,-1

        return points_lst_0,points_lst_1
    
    def _get_heatmap_tensors(self, dist, weight):
        grid = torch.tensor(
            [
                [
                    [x, y] for y in list(np.linspace(0, self.PI_max, self.PI_grid_num))
                ] for x in list(np.linspace(0, self.PI_max, self.PI_grid_num))
            ]
        ).to(torch.float32)
        heatmap_tensor_list = []
        for i in range(dist.shape[0]):
            points = self._get_pd_points(dist[i, :, :], weight[i, :])
            heatmap = PI_value(grid, points, h=self.PI_h, PI_weight=self.PI_weight)
            heatmap_tensor_list.append(heatmap)

        heatmap_tensors = torch.stack(heatmap_tensor_list, dim=0)
        return heatmap_tensors

    def forward(self, X):
        if self.input_format == "dist":
            dist = X.clone()
        else:
            dist = torch.cdist(X, X)
        
        # generate distance metrices corresponding to random subsets of point cloud 
        dist_new,dist_new_vec,lst_indices = sample_distance_matrices(dist, self.nb_repeat, self.subset_size)
        assert len(dist_new.shape) == 4,'Input error, no random distance !!'
        assert dist_new.shape[2] == dist_new.shape[3],'Input error, distance shape error !!' 

        nb_repeat = dist_new.shape[1]
        weight = torch.zeros((dist.shape[0], dist.shape[1]),
                             dtype=dist.dtype, device=dist.device)
        
        for i in range(nb_repeat):
            weight_tmp = self.weight_net(dist_new[:,i,:,:], dist_new_vec[:,i,:,:])
            weight = weight + weight_tmp

        weight = weight / nb_repeat

        assert self.perslay is not None, 'PersLay must not be None !!'

        # assign corresponding weights to each point in the subsampled point cloud
        sub_weight = torch.zeros((weight.shape[0], weight.shape[1],self.subset_size),
                                 dtype=weight.dtype, device=weight.device)
        
        assert len(lst_indices) == weight.shape[0], 'lst_indices error 1 !!'
        assert len(lst_indices[0]) == nb_repeat, 'lst_indices error 2 !!'

        for i in range(weight.shape[0]):
            for j in range(nb_repeat):
                sub_weight[i,j,:] = weight[i,lst_indices[i][j]]
        
        points_list = [sum([self._get_pd_points(dist_new[i,j,:,:], sub_weight[i,j,:]) \
                            for j in range(nb_repeat)],[]) for i in range(dist.shape[0])]
        
        representation = self.perslay(points_list)


        return representation
        
class Dist2Weight(nn.Module):
    def __init__(self, num_points,subset_size, **kwargs):
        super(Dist2Weight, self).__init__()
        self.num_points = num_points
        # print('num_points: '+str(num_points))
        # self.get_pc_feature = MatrixDeepSets(1, num_points, 16, doub=True, **kwargs)
        # self.get_point_feature = MatrixDeepSets(1, num_points, 8, doub=False, **kwargs)
        self.get_pc_feature = MatrixDeepSets(1, subset_size, subset_size, 16, doub=True, **kwargs)
        self.get_point_feature = MatrixDeepSets(1, num_points, subset_size, 8, doub=False, **kwargs)

        # self.device = kwargs.get("device", "cpu")
        self.device = 'cuda:0'

        self.get_pc_feature.to(self.device)
        self.get_point_feature.to(self.device)

        ### optional parameters ###
        # self.last_layer_bn = kwargs.get("last_layer_bn", False)
        self.last_layer_bn = True

        if self.last_layer_bn:
            print('Use batch norm in Dist2Weight')
        else:
            print('No batch norm in Dist2Weight')

        self.mlp_weight = nn.Sequential(
            NonLinear(24, 256), 
            NonLinear(256, 512), 
            NonLinear(512, 256), 
            NonLinear(256, 1, batch_norm=self.last_layer_bn)
            # nn.ReLU(inplace=True)
        )
    
    def forward(self, dist_mat, dist_vec):
        """
        Input
        dist_mat: batch_size x pc_point_num x pc_point_num
        dist_vec: batch_size x N x pc_point_num (N: number of points to calculate the weight)
        """

        # print(dist_mat.shape, dist_vec.shape)
        N = dist_vec.shape[1]

        dist_mat = dist_mat.unsqueeze(dim=3)

        # print(dist_mat.get_device())


        pc_feature= self.get_pc_feature(dist_mat) # batch_size x feature_dim

        h = torch.median(dist_vec,dim=-1,keepdim=True).values
        K_vec = torch.exp(-1*torch.pow(dist_vec/(h+0.1),2))

        DK_vec = K_vec.mean(dim=-1,keepdim=False)
        assert DK_vec.shape == (dist_vec.shape[0], N), 'DK_vec shape error'
        DK_max = DK_vec.mean(dim=-1,keepdim=True)
        DK_vec = DK_vec / DK_max

        dist_vec = dist_vec.unsqueeze(dim=3)
         
        point_feature = self.get_point_feature(dist_vec) # batch_size x N x feature_dim

        pc_feature_dup = torch.stack([pc_feature]*N, dim=1) # batch_size x N x feature_dim

        concat_feature = torch.cat([pc_feature_dup, point_feature], dim=2)

        out = concat_feature.view(-1, concat_feature.shape[-1])
        out = self.mlp_weight(out)
        out = out.view(-1, self.num_points, out.shape[-1])
        out_new = out[:,:,0]

        assert out_new.shape == DK_vec.shape, 'weight shape error: '+str(out.shape)+' vs '+str(DK_vec.shape)

        return  out_new * DK_vec

class Dist2WeightTransformer(nn.Module):
    def __init__(self, num_points, **kwargs):
        super(Dist2WeightTransformer, self).__init__()
        self.num_points = num_points
        ### optional parameters ###
        self.last_layer_bn = kwargs.get("last_layer_bn", False)

        self.get_point_feature = MatrixDeepSets(1, num_points, 32, doub=False, **kwargs)
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=4)
        self.fc = nn.Sequential(nn.Linear(32, 1), nn.ReLU(inplace=True))
    
    def forward(self, dist_mat):
        """
        Input
        dist_mat: batch_size x pc_point_num x pc_point_num
        dist_vec: *not used*
        """
        dist_mat = dist_mat.unsqueeze(dim=3)
        point_feature_seq = self.get_point_feature(dist_mat) # batch_size x N x feature_dim
        out = self.fc(self.transformer(point_feature_seq))
        return out

class PersLay(nn.Module):
    def __init__(self, num_labels, **kwargs):
        super(PersLay, self).__init__()
        ### optional parameters ###
        self.num_obsrv = kwargs.get("num_obsrv", 10)

        self.theta = nn.Parameter(torch.rand(self.num_obsrv, 2) * 4)
        self.fc = nn.Linear(self.num_obsrv, num_labels)
    
    def forward(self, points_list):
        h = 0.5
        out_list = []
        for L in points_list:
            if L:
                points_tensor = torch.stack([torch.stack(points, dim=0) for points in L], dim=0)
                exp_dist_to_points = torch.exp(- (torch.cdist(self.theta, points_tensor) ** 2) / (2*(h**2)))
                out_list.append(torch.sum(exp_dist_to_points, dim=1))
            else:
                out_list.append(torch.zeros(self.num_obsrv))
        out = torch.stack(out_list, dim=0)
        out = self.fc(out)   
        return out