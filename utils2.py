import torch

def sample_distance_matrices(distance_matrices, P, n):

    manual_seed = 42
    # torch.manual_seed(manual_seed)
    
    # Use idx to set seed
    M, N, _ = distance_matrices.shape
    sampled_matrices = torch.zeros((M, P, n, n), dtype=distance_matrices.dtype, device='cuda:0')
    sampled_matrices_vec = torch.zeros((M, P, N, n), dtype=distance_matrices.dtype, device='cuda:0')
    lst_indices = []

    for i in range(M):
        tmp_lst = []
        for j in range(P):
            torch.manual_seed(manual_seed+j)
            indices = torch.randperm(N)[:n]
            sampled_sub_matrix = distance_matrices[i, indices][:, indices]
            sampled_matrices[i, j] = sampled_sub_matrix
            sampled_sub_matrix_vec = distance_matrices[i, :][:,indices]
            sampled_matrices_vec[i, j] = sampled_sub_matrix_vec
            tmp_lst.append(indices)
        lst_indices.append(tmp_lst)

    return sampled_matrices,sampled_matrices_vec,lst_indices


def sample_distance_matrices_sub(distance_matrices, P, n):

    manual_seed = 42
    # torch.manual_seed(manual_seed)
    
    # Use idx to set seed
    distance_matrices = torch.from_numpy(distance_matrices)
    N, _ = distance_matrices.shape
    M = 1
    sampled_matrices = torch.zeros((M, P, n, n))
    # sampled_matrices_vec = torch.zeros((M, P, N, n), dtype=distance_matrices.dtype, device='cuda:0')
    # lst_indices = []

    for i in range(M):
        # tmp_lst = []
        for j in range(P):
            torch.manual_seed(manual_seed+j)
            indices = torch.randperm(N)[:n]
            sampled_sub_matrix = distance_matrices[indices,:][:, indices]
            sampled_matrices[i, j] = sampled_sub_matrix
            # sampled_sub_matrix_vec = distance_matrices[i, :][:,indices]
            # sampled_matrices_vec[i, j] = sampled_sub_matrix_vec
            # tmp_lst.append(indices)
        # lst_indices.append(tmp_lst)

    return sampled_matrices


# Permutauion Invariant Data Arguementation for point cloud / distance matrix data

def add_noise_to_distance_matrix(batch_distance_matrix, noise_scale=0.05):
    # add noise to distance matrix
    noise = torch.randn_like(batch_distance_matrix) * noise_scale
    noisy_batch_distance_matrix = batch_distance_matrix + noise
    for i in range(noisy_batch_distance_matrix.shape[0]):
        noisy_batch_distance_matrix[i] = (noisy_batch_distance_matrix[i] + noisy_batch_distance_matrix[i].T) / 2
        torch.diagonal(noisy_batch_distance_matrix[i]).fill_(0)
    return noisy_batch_distance_matrix

def scale_distance_matrix(batch_distance_matrix, scale_range=(0.5, 1.5)):
    # scale distance matrix with a random scale factor in scale_range
    scale_factor = torch.FloatTensor(1).uniform_(scale_range[0], scale_range[1])
    scale_factor = scale_factor.to('cuda:0')
    scaled_matrix = batch_distance_matrix * scale_factor
    return scaled_matrix



# Time-Delaying Embedding

import numpy as np
import pandas as pd
import os

def load_UCR_dataset(dataname,m=10):
    folder = 'UCRArchive_2018'+os.sep+dataname
    file_train = dataname+'_TRAIN.tsv'
    file_test = dataname+'_TEST.tsv'
    data_train = pd.read_csv(folder+os.sep+file_train,sep='\t',header=None)
    data_test = pd.read_csv(folder+os.sep+file_test,sep='\t',header=None)
    Y = np.array(list(data_train.values[:,0])+list(data_test.values[:,0]))
    Xs = np.concatenate([data_train.values[:,1:],data_test.values[:,1:]],axis=0)
    assert Xs.shape[0] == Y.shape[0], "X and Y have different number of samples"

    d = Xs.shape[1]
    X_lst = list()
    for X in Xs:
        point_cloud = list()
        for i in range(d):
            if i+m <= d:
                point = np.reshape(X[i:(i+m)],(1,-1))
                point_cloud.append(point)
        point_cloud = np.concatenate(point_cloud,axis=0)
        X_lst.append(torch.from_numpy(point_cloud))
    assert len(X_lst) == Y.shape[0], "X_lst and Y have different number of samples"

    X_ = torch.stack(X_lst,dim=0)
    print(np.unique(Y))
    Y_= torch.from_numpy(Y)
    # return X_lst, Y
    return X_,Y_


# if __name__ == '__main__':
    # data_lst = ['CinCECGTorso','InlineSkate','Mallat','StarLightCurves','HandOutlines','Phoneme','UWaveGestureLibraryAll']
    # X,Y = load_UCR_dataset('CinCECGTorso',m=10)
    # X,Y = load_UCR_dataset('InlineSkate',m=10)
    # X,Y = load_UCR_dataset('Mallat',m=10)
    # X,Y = load_UCR_dataset('StarLightCurves',m=10)
    # X,Y = load_UCR_dataset('HandOutlines',m=10)
    # X,Y = load_UCR_dataset('Phoneme',m=10)
    # X,Y = load_UCR_dataset('UWaveGestureLibraryAll',m=10)
    # selected_dname = 'HandOutlines'
    # selected_dname = 'Earthquakes'
    # X,Y = load_UCR_dataset(selected_dname,m=10)
    # print(X.shape,Y.shape)
    # print(type(X.data),type(Y.data),Y)

    # for dname in data_lst:    
    #     X,Y = load_UCR_dataset(dname,m=10)

    #     print(dname,' : ',X.shape)


