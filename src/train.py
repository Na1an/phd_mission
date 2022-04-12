import torch
import numpy as np

class Trainer():
    #def __init__(self, model, device, train_data, val_data, opt="Adam"):
    def __init__(self, model, device, opt="Adam"):
        # put our data to device
        self.device = device
        self.model = model.to(device)
        
        # optimizer
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # take data
        #self.train_data = train_data
        #self.val_data = val_data

    def train_cuboid(self, points, voxel_skeleton, nb_query_point, local_epoch=20):
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
        for e in range(local_epoch):
            print(e)

        print("===========" + "The cuboid {}/{} is trained.".format(1,10) +"===========")

    def train_model(self, train_data, voxel_skeleton, nb_epoch=20):
        '''
        Args:
            nb_epoch : a integer. How many epochs you want to train.
            train_data : a np.darray. (x, y, z, label)

        Return:
        '''
        # initialize loss
        loss = 0

        for i in range(0, nb_epoch):
            # batch.size = n
            # take n iterations to complete an epoch
            sum_loss = 0
            print("start {}-th epoch".format(i))




        return None