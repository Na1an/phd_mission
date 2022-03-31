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