
# Training datapart 

## 1. Duplicate points for training
```
https://github.com/charlesq34/pointnet/issues/161

charlesq34 commented on Sep 17, 2019
HI @brbrhpt

As there are batch norm layers, using batch size = 1 would cause some issues. At training time, TensorFlow does require a fixed tensor size, but you can achieve the "effect" of training with variable number of points by randomly duplicating K existing points to N points (e.g. K is varying, N is fixed as 1024)-- for pointnet, the results would be roughly the same (just a difference caused by batchnorm) as training with K points.

At test time by setting batch size as 1, you can always test with variable number of points.

@charlesq34
I tried training by duplicating the points and it works but not sure if it works roughly the same when testing on variable length. I want to ask to be sure about mean/std normalization of model input.

When training I have fixed 100 points (usually my dataset have around 40-70 points). So the extra points are padded by duplications.
When testing I want to test with 40-70 points not 100 points.

Suppose that I do standard scaling for each point cloud on the X Y Z coordinate, should I do it before or after I duplicate the points?

If I duplicate before I normalize, this will affect the coordinate values. But maybe it will make the model generalize better by not relying on specific coordinate value?
If I duplicate after I normalize, this will be like I'm just appending extra points when training. I'm not sure which approach is better.
Please enlighten me.

--------
Hi @off99555 I also want to know this. But i think, first copy the points and then normalize will be better because at end we need scaled or normalized data.

--------

@Maheshiitmandi But normalizing before duplications is also having its benefits. Suppose you have 70 points and you want to expand to 100 points, if you normalize before duplicates you will get the same prediction from the model (if you are doing classification), because most of the layers just do max pooling. And same prediction for 70 points and 100 points is what you want because the extra 30 points are just for padding.
```

## rigid sliding window -> soft sliding window

* Instead of dividing the data, select randomly the coordinates of the window. No limit on traning data.


# Model part

## Pointnet++ should be better than pointnet