import sys
import random
import time
import datetime
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from gudhi.dtm_rips_complex import DTMRipsComplex

from FiltrationLearningForPointClouds.scripts.lib.pc_representation import *
from FiltrationLearningForPointClouds.scripts.lib.pointnet import *
from FiltrationLearningForPointClouds.scripts.lib.scheduler import *
from expected_toporep_2 import ExpTopoRep
import time
import os
from utils2 import *
from early_stopping import EarlyStopping


import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    start_time = time.time()
    args = sys.argv[1:]
    print(args)
    if not args:
        args = ["result", "KNproteinNoisy01_C=7,T=500,K=60", 0, 1, 0, 200]

    ### Save Directory ###
    savedirname = args[0]
    ### Dataset ###
    dataset = args[1]
    ### Network Architecture ###
    rips = int(args[2])
    toporep = int(args[3])
    dtm = int(args[4])
    nb_repeat = int(args[5])
    dim = int(args[6])
    bs = int(args[7])

    pointnet = int(args[8])
    deepsets = int(args[9])
    pointmlp = int(args[10])

    if rips:
        print('Rips Filtration')
    elif dtm:
        print('DTM Filtration')
    elif toporep:
        print('Filtration Learning via PPM')
    else:
        assert False,'WTF'

    ### Optimization Hyper Parameters ###
    epoch_num = 200 
    warmup_epoch_num = 40
    batch_size = bs
    lr =  1e-1
    print('Leanring Rate = ',lr)
    ### Regularization ###
    lamb = -1 # regularization on perslay
    reg_ord = 1
    if lamb<0:
        print('No regularization on perslay')
    ### Optional Parameters ###
    CV_num = 3
    CV_color_list = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    method_toporep = "dist"
    # always transform input format to distance
    input_format = "dist"
    all_X = torch.load(f"FiltrationLearningForPointClouds/data/{dataset}_data").to(torch.float32)
    all_X = pointcloud_normalize(all_X[:, :128, :])
    # print(all_X.shape)
    # all_X = torch.cdist(all_X,all_X)
    all_y = torch.load(f"FiltrationLearningForPointClouds/data/{dataset}_label")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    device = "cuda:0"
    print(f"Device: {device}")
    # all_X = all_X.to(device)
    # all_y = all_y.to(device)

    data_num = all_X.shape[0]
    num_points = all_X.shape[1]
    output_dim = 10
    criterion = nn.CrossEntropyLoss()
    task = "cls"

    # all_X = torch.cdist(all_X,all_X)
    # all_X_new,all_X_vec,lst_indices = sample_distance_matrices(all_X, nb_repeat, 4)


    ### 5-fold Cross Validation ###
    epoch_num_0 = 1000
    valid_task_loss_list = [[0]*epoch_num_0 for _ in range(CV_num)]
    if task == "cls": 
        valid_task_acc_list = [[0]*epoch_num_0 for _ in range(CV_num)]
        test_task_acc_list = [[0]*epoch_num_0 for _ in range(CV_num)]
    data_idx = list(range(data_num))
    # data_idx.sort(key=lambda i: (all_y[i], i))
    random.seed(2026)
    random.shuffle(data_idx)
    input_dim = all_X.shape[2]

    # for CV_idx in range(CV_num):
    #         print(f"--- Cross Validation: {CV_idx + 1} / {CV_num} ---")
    #         train_idx = [data_idx[i] for i in range(data_num) if i % CV_num != CV_idx]
    #         # train_idx = train_idx[200:]
    #         trainX = all_X[train_idx, :, :]
    #         trainy = all_y[train_idx]
    #         valid_idx = [data_idx[i] for i in range(data_num) if i % CV_num == CV_idx]
    #         validX = all_X[valid_idx, :, :]
    #         validy = all_y[valid_idx]
    #         train_num = trainX.shape[0]
    #         valid_num = validX.shape[0]
    #         task_output_dim = 10

    #         if task == "cls":
    #             print("Train Data Distribution: ", {j: int(sum([trainy[i] == j for i in range(train_num)])) for j in range(task_output_dim)})
    #             print("Valid Data Distribution: ", {j: int(sum([validy[i] == j for i in range(valid_num)])) for j in range(task_output_dim)})

    #         input_dim = all_X.shape[2]
    #         first_feature_dim = 16 * (pointnet + deepsets + pointmlp)
    #         first_feature_net = PCFeatureNet(input_dim, num_points, pointnet=pointnet, deepsets=deepsets, pointmlp=pointmlp)

            
    #         if task == "cls":
    #             first_task_solver = nn.Sequential(
    #                 nn.Linear(first_feature_dim, task_output_dim), 
    #                 nn.Softmax(dim=1),
    #             )
    #         else:
    #             first_task_solver = nn.Sequential(
    #                 nn.Linear(first_feature_dim, task_output_dim),
    #             )
                
    #         first_feature_net.eval()
    #         first_task_solver.eval()
    #         with torch.no_grad():
    #             valid_out = first_task_solver(first_feature_net(validX))
    #             valid_loss = float(criterion(valid_out, validy))
    #             if task == "cls":
    #                 valid_acc = 100 * float(sum(torch.max(valid_out, dim=1).indices == validy) / valid_num)
    #                 print(f"Initial valid_loss = {valid_loss:.3f}, Initial valid_acc = {valid_acc:.1f}")
    #             else:
    #                 print(f"Initial valid_loss = {valid_loss:.3f}")
            
    #         first_warmup_epoch_num = 40
    #         first_lr = 1e-2 if pointmlp else 2e-2
    #         first_opt = torch.optim.Adam(list(first_feature_net.parameters()) + list(first_task_solver.parameters()), lr=first_lr)
    #         first_scheduler = TransformerLR(first_opt, warmup_epochs=first_warmup_epoch_num) if first_warmup_epoch_num is not None else None
    #         train_toporep_points_list = [[] for _ in range(trainX.shape[0])] if toporep else None
    #         for epoch in range(epoch_num_0):
    #             first_feature_net.train()
    #             first_task_solver.train()
                    
    #             epoch_loss_list = []
    #             idx_list = random.sample(range(train_num), train_num)
    #             for idx in range(train_num//batch_size):
    #                 first_opt.zero_grad()
    #                 batch_idx_list = idx_list[batch_size * idx: batch_size * (idx+1)]
    #                 batchX = random_rotation(trainX[batch_idx_list, :, :])
    #                 batch_out = first_task_solver(first_feature_net(batchX))
    #                 batch_loss = criterion(batch_out, trainy[batch_idx_list])
    #                 batch_loss.backward()
    #                 first_opt.step()
    #                 epoch_loss_list.append(float(batch_loss))
                
    #             first_feature_net.eval()
    #             first_task_solver.eval()
    #             with torch.no_grad():
    #                 valid_out = first_task_solver(first_feature_net(validX))
    #                 valid_loss = float(criterion(valid_out, validy))
    #                 if task == "cls":
    #                     valid_acc = 100 * float(sum(torch.max(valid_out, dim=1).indices == validy) / valid_num)
    #                     print(f"Epoch {epoch}: average_loss = {sum(epoch_loss_list)/len(epoch_loss_list):.3f}, valid_loss = {valid_loss:.3f}, valid_acc = {valid_acc:.1f}", flush=True)
    #                 else:
    #                     print(f"Epoch {epoch}: average_loss = {sum(epoch_loss_list)/len(epoch_loss_list):.3f}, valid_loss = {valid_loss:.3f}", flush=True)
                    
    #                 valid_task_loss_list[CV_idx][epoch] = valid_loss
    #                 if task == "cls": valid_task_acc_list[CV_idx][epoch] = valid_acc
            
    #             if first_scheduler is not None: 
    #                 first_scheduler.step()
            
    #         torch.save(first_feature_net.state_dict(), f"{savedirname}/first_feature_net_CVidx={CV_idx}.pth")
    #         torch.save(first_task_solver.state_dict(), f"{savedirname}/first_task_solver_CVidx={CV_idx}.pth")
    #         dnn_pretrained_model_path = savedirname
    

    # res = np.array([valid_task_loss_list[i][-1] for i in range(CV_num)])
    # print('PointNet: ',np.mean(res),np.std(res))

    if pointnet:
        dnn_pretrained_model_path = 'premodel'+os.sep+'pointnet'
    elif deepsets:
        dnn_pretrained_model_path = 'premodel'+os.sep+'deepsets'
    elif pointmlp:
        dnn_pretrained_model_path = 'premodel'+os.sep+'pointmlp'
    else:
        assert False,'Wrong Pre Model'

    all_X_ori = all_X.clone()
    all_y_ori = all_y.clone()

    all_X = torch.cdist(all_X,all_X)
    all_X_new,all_X_vec,lst_indices = sample_distance_matrices(all_X, nb_repeat, 4)

    valid_task_loss_list = [[0]*epoch_num for _ in range(CV_num)]
    if task == "cls": 
        valid_task_acc_list = [[0]*epoch_num for _ in range(CV_num)]
        test_task_acc_list = [[0]*epoch_num for _ in range(CV_num)]
    
    for CV_idx in range(CV_num):
        early_stopping = EarlyStopping(save_path=savedirname, patience=30, verbose=True,delta=0.001)
        print(f"=== Cross Validation: {CV_idx + 1} / {CV_num} ===")
        all_y = all_y.to(device)

        train_idx = [data_idx[i] for i in range(data_num) if i % CV_num != CV_idx]

        valid_idx = train_idx[:200]
        train_idx = train_idx[200:]

        # trainning set
        trainX = all_X_new[train_idx, :, :,:]
        trainX_vec = all_X_vec[train_idx,:,:,:]
        trainy = all_y[train_idx]
        train_assign_ids = [lst_indices[i] for i in train_idx]

        # validation set
        validX = all_X_new[valid_idx, :,:,:]
        valid_vec = all_X_vec[valid_idx,:,:,:]
        validy = all_y[valid_idx]
        valid_assign_ids = [lst_indices[i] for i in valid_idx]

        # testing set
        test_idx = [data_idx[i] for i in range(data_num) if i % CV_num == CV_idx]
        testX = all_X_new[test_idx, :,:,:]
        test_vec = all_X_vec[test_idx,:,:,:]
        testy = all_y[test_idx]
        test_assign_ids = [lst_indices[i] for i in test_idx]

        # validation set Ori

        validX_ori = all_X_ori[valid_idx,:,:]
        validy_ori = all_y_ori[valid_idx]

        # testing set Ori

        testX_ori = all_X_ori[test_idx,:,:]
        testy_ori = all_y_ori[test_idx]

        # trainning set Ori

        trainX_ori = all_X_ori[train_idx,:,:]
        trainy_ori = all_y_ori[train_idx]



        train_num = trainX.shape[0]
        valid_num = validX.shape[0]
        test_num = testX.shape[0]

        first_model_name = 'point network'
        # input_dim = all_X.shape[2]
        num_points = 128
        if first_model_name is not None:
            first_feature_net = PCFeatureNet(input_dim, num_points, pointnet=pointnet, deepsets=deepsets, pointmlp=pointmlp)
            first_feature_net.load_state_dict(torch.load(f"{dnn_pretrained_model_path}/first_feature_net_CVidx={CV_idx}.pth"))
            for param in first_feature_net.parameters(): param.requires_grad = False
            first_feature_net.eval()

        if task == "cls":
            print("Train Data Distribution: ", {j: int(sum([trainy[i] == j for i in range(train_num)])) for j in range(output_dim)})
            print("Test Data Distribution: ", {j: int(sum([validy[i] == j for i in range(valid_num)])) for j in range(output_dim)})

        feature_dim = 16
        if toporep:     
            toporep_net = ExpTopoRep(1, 1, 
                                  reducer=False, perslay=False, method='dist', 
                                  input_format="dist")
            for param in toporep_net.weight_net.parameters(): torch.nn.init.normal_(param, mean=0.0, std=1)
        feature_net = PCFeatureNet(None, num_points, matds_dist=0, ph=(dtm or toporep))

        if dim<0:
            # Use multidimensional PPM
            task_solver = nn.Sequential(
                nn.Linear(feature_dim*2+16 * (pointnet + deepsets + pointmlp), output_dim), 
                nn.Softmax(dim=1),
            )
        else:
            # Use PPM of a specfic dimension
            task_solver = nn.Sequential(
                nn.Linear(feature_dim+16 * (pointnet + deepsets + pointmlp), output_dim), 
                nn.Softmax(dim=1),
            )

        toporep_net.to(device)
        feature_net.to(device)
        task_solver.to(device)


        # for param in task_solver.parameters():
        #     param.requires_grad = False
        # for para in feature_net.perslay.parameters():
        #     param.requires_grad = False
        
        feature_net.eval()
        task_solver.eval()

        if toporep: toporep_net.eval()
        with torch.no_grad():
            if dtm:
                # valid_points_list_0 = valid_dtm_points_list
                assert False
            elif toporep:
                valid_points_list_0,valid_points_list_1 = toporep_net.get_PPM_rips(validX,valid_vec,valid_assign_ids)
            else:
                valid_points_list = None

            if dim == 0:
                feature_out = feature_net(validX, pd_points_list=valid_points_list_0)
            elif dim == 1:
                feature_out = feature_net(validX, pd_points_list=valid_points_list_1)
            else:
                feature_perslay_0 = feature_net(validX, pd_points_list=valid_points_list_0)
                feature_perslay_1 = feature_net(validX, pd_points_list=valid_points_list_1)
                feature_out = torch.cat((feature_perslay_0,feature_perslay_1),dim=1)

            feature_out = torch.cat((feature_out,first_feature_net(validX_ori).to(device)),dim=1)
            valid_out = task_solver(feature_out)

            valid_loss = float(criterion(valid_out, validy))

            test_points_list_0,test_points_list_1 = toporep_net.get_PPM_rips(testX,test_vec,test_assign_ids)

            if dim == 0:
                feature_out = feature_net(testX, pd_points_list=test_points_list_0)
            elif dim == 1:
                feature_out = feature_net(testX, pd_points_list=test_points_list_1)
            else:
                feature_perslay_0 = feature_net(testX, pd_points_list=test_points_list_0)
                feature_perslay_1 = feature_net(testX, pd_points_list=test_points_list_1)
                feature_out = torch.cat((feature_perslay_0,feature_perslay_1),dim=1)
            
            feature_out = torch.cat((feature_out,first_feature_net(testX_ori).to(device)),dim=1)
            
            test_out = task_solver(feature_out)
            test_loss = float(criterion(test_out, testy))

            if task == "cls":

                valid_acc = 100 * float(sum(torch.max(valid_out, dim=1).indices == validy) / valid_num)
                print(f"Initial valid_loss = {valid_loss}, Initial valid_acc = {valid_acc}")

                test_acc = 100 * float(sum(torch.max(test_out, dim=1).indices == testy) / test_num)
                print(f"Initial valid_loss = {valid_loss}, Initial valid_acc = {valid_acc}")
            else:
                print(f"Initial valid_loss = {valid_loss}")
        
        # opt = torch.optim.Adam(list(toporep_net.parameters()) + list(feature_net.parameters()) + list(task_solver.parameters()), lr=lr)
        opt = torch.optim.Adam(list(feature_net.parameters()) + list(task_solver.parameters()), lr=lr)

        scheduler = TransformerLR(opt, warmup_epochs=warmup_epoch_num) if warmup_epoch_num is not None else None

        for epoch in range(epoch_num):
            feature_net.train()
            task_solver.train()
            if toporep: toporep_net.train()

            acc_list = []

            epoch_loss_list = []
            idx_list = random.sample(range(train_num), train_num)
            for idx in range(train_num//batch_size):
                opt.zero_grad()
                batch_idx_list = idx_list[batch_size * idx: batch_size * (idx+1)]
                batchX = trainX[batch_idx_list, :, :,:]
                batchX_ori = trainX_ori[batch_idx_list,:,:]
                batchVec = trainX_vec[batch_idx_list, :, :,:]
                batch_ids = [train_assign_ids[i] for i in batch_idx_list]
                if dtm:
                    # batch_points_list_0 = [train_dtm_points_list[i] for i in batch_idx_list]
                    assert False
                elif toporep:
                    # batch_points_list = toporep_net.get_PPM(batchX)
                    batch_points_list_0,batch_points_list_1= toporep_net.get_PPM_rips(batchX,batchVec,batch_ids)
                else:
                    batch_points_list = None
                
                if dim == 0:
                    feature_out = feature_net(batchX, pd_points_list=batch_points_list_0)
                elif dim == 1:
                    feature_out = feature_net(batchX, pd_points_list=batch_points_list_1)
                else:
                    feature_perslay_0 = feature_net(batchX, pd_points_list=batch_points_list_0)
                    feature_perslay_1 = feature_net(batchX, pd_points_list=batch_points_list_1)
                    feature_out = torch.cat((feature_perslay_0,feature_perslay_1),dim=1)

                feature_out = torch.cat((feature_out,first_feature_net(batchX_ori).to(device)),dim=1)
                batch_out = task_solver(feature_out)
                
                train_acc = 100 * float(sum(torch.max(batch_out, dim=1).indices == trainy[batch_idx_list]) / batchX.shape[0])
                acc_list.append(train_acc)
                batch_loss = criterion(batch_out, trainy[batch_idx_list])

                if lamb > 0:
                    if feature_net.perslay is None: 
                        raise Exception("PersLay is not used. ")
                    batch_loss += lamb * torch.linalg.norm(feature_net.perslay.fc[0].weight, ord=reg_ord)

                batch_loss.backward()
                opt.step()
                epoch_loss_list.append(float(batch_loss))
            
            feature_net.eval()
            task_solver.eval()
            if toporep: toporep_net.eval()
            with torch.no_grad():
                if dtm:
                    # valid_points_list_0 = valid_dtm_points_list
                    assert False
                elif toporep:
                    # valid_points_list = toporep_net.get_PPM(validX)
                    valid_points_list_0,valid_points_list_1 = toporep_net.get_PPM_rips(validX,valid_vec,valid_assign_ids)
                else:
                    valid_points_list = None
                # valid_out = task_solver(feature_net(validX, pd_points_list=valid_points_list))
                if dim == 0:
                    feature_out = feature_net(validX, pd_points_list=valid_points_list_0)
                elif dim == 1:
                    feature_out = feature_net(validX, pd_points_list=valid_points_list_1)
                else:
                    feature_perslay_0 = feature_net(validX, pd_points_list=valid_points_list_0)
                    feature_perslay_1 = feature_net(validX, pd_points_list=valid_points_list_1)
                    feature_out = torch.cat((feature_perslay_0,feature_perslay_1),dim=1)

                feature_out = torch.cat((feature_out,first_feature_net(validX_ori).to(device)),dim=1)
                
                valid_out = task_solver(feature_out)

                valid_loss = float(criterion(valid_out, validy))

                test_points_list_0,test_points_list_1 = toporep_net.get_PPM_rips(testX,test_vec,test_assign_ids)

                if dim == 0:
                    feature_out = feature_net(testX, pd_points_list=test_points_list_0)
                elif dim == 1:
                    feature_out = feature_net(testX, pd_points_list=test_points_list_1)
                else:
                    feature_perslay_0 = feature_net(testX, pd_points_list=test_points_list_0)
                    feature_perslay_1 = feature_net(testX, pd_points_list=test_points_list_1)
                    feature_out = torch.cat((feature_perslay_0,feature_perslay_1),dim=1)
                
                feature_out = torch.cat((feature_out,first_feature_net(testX_ori).to(device)),dim=1)
                test_out = task_solver(feature_out)
                test_loss = float(criterion(test_out, testy))

                if task == "cls":
                    valid_acc = 100 * float(sum(torch.max(valid_out, dim=1).indices == validy) / valid_num)
                    test_acc = 100 * float(sum(torch.max(test_out, dim=1).indices == testy) / test_num)
                    train_acc = sum(acc_list)/(train_num//batch_size)
                    print('Train acc = ',train_acc)
                    print(f"Epoch {epoch}: average_loss = {sum(epoch_loss_list)/len(epoch_loss_list):.3f}, valid_loss = {valid_loss:.3f}, valid_acc = {valid_acc:.1f}, test_loss = {test_loss:.3f}, test_acc = {test_acc:.1f}", 
                          flush=True)
                else:
                    print(f"Epoch {epoch}: average_loss = {sum(epoch_loss_list)/len(epoch_loss_list):.3f}, valid_loss = {valid_loss:.3f}", flush=True)

                valid_task_loss_list[CV_idx][epoch] = valid_loss
                if task == "cls": 
                    valid_task_acc_list[CV_idx][epoch] = valid_acc
                    test_task_acc_list[CV_idx][epoch] = test_acc
                
                models = [feature_net,task_solver,toporep_net]
                early_stopping(valid_loss, valid_acc, models,CV_idx,savedirname,test_acc)

                if early_stopping.early_stop:
                    print("Early Stopping")
                    print('Final Acc On Test = ',early_stopping.best_acc)
                    test_task_acc_list[CV_idx][-1] = early_stopping.best_acc
                    break

            if scheduler is not None:
                scheduler.step()
                
        if not early_stopping.early_stop:
            test_task_acc_list[CV_idx][-1] = early_stopping.best_acc
            
            # for name, param in toporep_net.named_parameters():
            #     if param.grad is not None:
            #         print(f'parameter name : {name}, grad: {param.grad}')
            #     else:
            #         print(f'parameter name: {name}, is None')

        # torch.save(feature_net.state_dict(), f"{savedirname}/feature_net_CVidx={CV_idx}.pth")
        # torch.save(task_solver.state_dict(), f"{savedirname}/task_solver_CVidx={CV_idx}.pth")
        # if toporep: torch.save(toporep_net.state_dict(), f"{savedirname}/toporep_CVidx={CV_idx}.pth")

    print(f"=== Summary ===")
    print(f"Time consuming: {datetime.timedelta(seconds=time.time() - start_time)}")
    avg = {}
    std = {}
    if task == "cls":
        avg["FinalAccuracy"] = np.mean([test_task_acc_list[i][-1] for i in range(CV_num)])
        std["FinalAccuracy"] = np.std([test_task_acc_list[i][-1] for i in range(CV_num)], ddof=0)
        print("Distribution of Final Accuracy", [f"{float(test_task_acc_list[i][-1]):.1f}" for i in range(CV_num)])
    
    avg["FinalLoss"] = np.mean([valid_task_loss_list[i][-1] for i in range(CV_num)])
    std["FinalLoss"] = np.std([valid_task_loss_list[i][-1] for i in range(CV_num)], ddof=0)
    print("Distribution of Valid Loss", [f"{float(valid_task_loss_list[i][-1]):.3f}" for i in range(CV_num)])

    for k in avg.keys():
        _avg = avg[k]
        _std = std[k]
        print(f"Average {k}: {_avg}")
        print(f"Std of {k}: {_std}")
        print(f"{_avg:.3f} ± {_std:.3f}")
        print(f"{_avg:.2f} ± {_std:.2f}")
        print(f"{_avg:.1f} ± {_std:.1f}")
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(CV_num):
        ax.plot(list(range(epoch_num)), valid_task_loss_list[i], color=CV_color_list[i])
    fig.savefig(f"{savedirname}/loss_history.png")
    if task == "cls":
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(CV_num):
            ax.plot(list(range(epoch_num)), test_task_acc_list[i], color=CV_color_list[i])
        fig.savefig(f"{savedirname}/accuracy_history.png")

    with open(f"{savedirname}/loss_history", "wb") as f:
        pickle.dump(valid_task_loss_list, f)
    if task == "cls":
        with open(f"{savedirname}/accuracy_history", "wb") as f:
            pickle.dump(test_task_acc_list, f)