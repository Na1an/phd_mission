# Development notes



## 31/03/22

* Try grid_sample, if-net (?)
  * grid_sample is not a method for voxelization, but a interpolation method
    * rich the detail (resolution/density) of the raw data 
    * fix the input size
  * sliding window + voxelization, net-work prefers the fixed sized input : so voxels are better than the point
  * implement steps **(1) preprocess_data + (2) sliding_window (local minimum position) + (3) voxelization**
* **[important]** grid_sample probleme solved
  * grid_sample is still a point-wise method, one step feature extraction is performed, in the form of voxels, but still notice that we always feed the network with the points, not voxels. **Even I would like to name this feature extraction step : voxelization feature extraction, the study object is still points.**
    * This makes us to find a way to subsample the point-clouds in a cube
    * **query_points** is a set of points selected randomly in the input (points), grid_sample(input, **query_points**)
      * the benefit of the query points is the implicit representation, we can put any point size as input, 10w ~ 100w, but we then use **grid_sample** to resize the input. That's allow us to feed the network. 
      * 100w points as input, select randomly 10000 query_points, which can represent the whole and then the output of the network is also 100w, each point will be given a class leaf/wood.   

## 23/08/23

* multiple GPU on one cluster

  * [not sure] Nvidia 2080 ti is different from V100 or A100, the RAM of 2080 series (consumer-grade product) will be merged automatically, the V100 and A100 are differerent.  

  * [sure] Need to use ```nn.DataParallel``` or ```torch.utils.data.distributed.DistributiedSampler``` 

    * see official example [here](https://pytorch.org/tutorials/intermediate/dist_tuto.html#distributed-training)

      