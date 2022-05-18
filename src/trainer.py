import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

class Trainer():
    def __init__(self, model, device, train_dataset, voxel_nets, batch_size, num_workers, shuffle=True, opt="Adam"):
        '''
        Args:
            model: the Deep Learning model.
            train_dataset: the dataset we need.
            voxel_skeleton: 
            batch_size: a integer.
            num_workers: a integer.
            shuffle: shuffler or not.
            opt: optimizer.
        '''
        # put our data to device & DataLoader
        self.device = device
        self.model = model.to(device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.voxel_nets = torch.from_numpy(voxel_nets.copy()).type(torch.float).to(self.device)
        
        # optimizer
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # take data
        #self.train_data = train_data
        #self.val_data = val_data
    
    def train_model(self, nb_epoch=200):
        '''
        Args:
            nb_epoch : a integer. How many epochs you want to train.
            train_data : a np.darray. (x, y, z, label)

        Return:
        '''
        # initialize loss

        #self.optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        epoch_loss=0 #compute_loss()
        epoch_acc=0

        for e in range(nb_epoch):
            self.model.train()
            running_loss = 0.0
            running_acc = 0
            i = 0
            # points, labels, v_cuboid
            for p, l, v in self.train_loader:
                print(p.shape)
                print("what is v={}".format(self.voxel_nets[v]))
                output = self.model(p, self.voxel_nets[v])

                print("=====================lalal==================")
                exit()
            
            print("============ Epoch {}/{} is trained ===========".format(e, nb_epoch))
        return None

    
    def __train_cuboid(self, points, voxel_skeleton, nb_query_point, epoch=200):
        '''
        Args:
            points : a np.darray. The points in a cuboid.(x, y, z, label).
            voxel_skeleton : a 4-D np.darray. (x, y, z, 1/0), occupied voxel.
            nb_query_point: a integer. The number of query points.
            local_epoch : a integer. How many epochs you want to train.

        Return:
            None.
        '''
        if nb_query_point > len(points):
            raise Exception("!! Inside train_cuboid, nb_query_point > len(points)")
        
        self.model.train()
        self.optimizer.zero_grad()
        cuboid_loss=0 #compute_loss()
        
        cuboid_loss.backward()
        self.optimizer.step()

        points_t = torch.from_numpy(points).to(self.device)
        voxel_skeleton_t = torch.from_numpy(voxel_skeleton).to(self.device)
        for e in range(epoch):
            print("lala")
            #for data in train_data:
            #    print("lala")

            

        print("===========" + "The cuboid {}/{} is trained.".format(1,10) +"===========")