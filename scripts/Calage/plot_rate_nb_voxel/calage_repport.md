# Registration criteria for phenOBS data

Yuchen, 20/04/22.



## 1. Digital Terrain Model

For dealing with the registration problem, the first step is to separate the **DTM** (Digital Terrain Model). It has a variety of forms, slope, rough and even with gaps. Especially we need to check if there are holes in it. The appearance of holes will make us lose part of the ground information. In the next step we will separate layers that are parallel from the DTM at different heights. If there are holes we have to do an interpolation parallel to the surrounding lands. 

So I suggest to use TLS DTM which is more dense than DLS DTM to make sure we have a more solid base. In the other hand, TLS data is closer to the ground and has more accurate and detailed details, which is also a natural reason to chose it.

<img src = "/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/image/tls_dtm.png" align="center" style="zoom:60%">

<center>Fig-1. TLS DTM</center>

<img src = "/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/image/dls_dtm.png" align="center" style="zoom:60%">

<center>Fig-2. DLS DTM (point size increased)</center>

DTM will be used as a reference to obtain layers like this:

<img src = "/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/image/exp_para_to_dtm.png" align="center" style="zoom:60%">

<center>Fig-3. Layers parallel to DTM</center>



## 2. Three different approaches

The next task is to find a criteria for the registration quality. TLS data is denser towards the bottom, DLS data is opposite. To find a compromise between both, so that we can get enough local details at a suitable height is a challenge. I got some help from the first approach : **Calculate Voxel Coincidence Rate**. The second approach, **Calculate Barycenter Distance**, also relies on the observations provided by the first one (not 100%). The last approach, **Calculate Canopy Height Difference**, is completely independent of DTM or different parallel layers.



### 2.1 Voxel Coincidence Rate

We will first voxelize TLS and DLS data :

<center class="half">
	<img src="/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/image/tls_voxelized_data.png" width="450">
	<img src="/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/image/dls_voxelized_data.png" width="450">  
</center>

<center>Fig-4. Voxelized TLS and DLS data, voxel_size=0.1m </center>

Note that above is a special case, it only shows a part of identified data, **the color = nb of points in one voxel/max(nb of points in all voxel)**. We will voxelize layer which looks like Fig-3.

The next step is to voxelize the layer for both data at the same relative position and find the coincident voxel (a voxel that is occupied in both TLS and DLS data). Then we can get two plots, on a data with side length equals 18m, starting from 2m above the ground and going up to 40m. Slice a layer every meter, voxelize and calculate the coincidence rate. 

* Here is coincidence rate plot:

<img src = "/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/coi_rate_sh_1m_vs_01m_gs_18m.png" align="center" style="zoom:60%">

<center>Fig-5. coincidence rate in different relative height, voxel_size=0.1m, slice_height=1m</center>

* We can also have a plot about voxel number:

  <img src = "/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/nb_voxel_sh_1m_vs_01m_gs_18m.png" align="center" style="zoom:60%">

  <center>Fig-6. voxel number in different relative height, voxel_size=0.1m, slice_height=1m </center>

* We can conclude that **coincidence rate** reaches the maximum between 26 and 33 meters above the ground. A variety of different scenarios have been tested: voxel_size=0.05m, 0.1m, 0.2m; slice_height=1m and 2m etc...

  * Increasing the **voxel size** will increase the coincidence rate

  * Increase the **slice height** has no big effect

* But no matter how the parameters are changed, the max value is always obtained between 28m and 33m above the ground, this may be a useful empirical parameter for us.

### 2.2 Barycenter/centroid distance

In case we have found the **highest coincidence rate** at 26m above the ground, this height is used to calculate the barycenter distance. We will use the **DBSCAN** (density-based spatial clustering of applications with noise) algorithm to assign the points in each layer into different clusters.

<center class="half">
	<img src="/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/image/capture_bis.png" width="450">
	<img src="/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/image/capture_dls.png" width="450">  
</center>

<center>Fig-7. Cluster A B C in TLS and DLS data </center>

* Once we have explicit clusters, then we can calculate the centroid distance between clusters

  <center class="half">
  	<img src="/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/image/capture_distance.png" width="450">
  	<img src="/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/image/capture_distance_258.png" width="450">  
  </center>

  <center>Fig-8. The position of A B C cluster at 25.6m~25.8m and 25.8m~26.0m </center>

  | cluster | layer = 25.6m ~ 25.8m | layer = 25.6m ~ 25.8m |
  | :-----: | :-------------------: | :-------------------: |
  |    A    |   0.356301936833279   |  0.3352485546807909   |
  |    B    |  0.12971863415012552  |  0.1496607768914889   |
  |    C    |  0.23760388866575213  |  0.3287834928973832   |

<center>Tab-1. The centroid distance of A B C clusters under different slice heights </center>

* The distance between the trunks is very obvious, so it means the registration still can be improved. 



### 2.3 Canopy height difference

DLS data is obtained from top to bottom, so the shape of the tree canopy is well preserved. If we believe that TLS data can keep canopy shape well (be able to scan top leaves), it is a good way to divide the data into small blocks and compare the difference of local maximum point height.

* We could compare the canopy local height difference between DLS (target1) data and TLS (target2) data:

<img src = "/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/chd_tls_dls.png" align="center" style="zoom:100%">

<center>Fig-9. DLS, TLS, TLS-DLS, resolution=0.1m, the colorbar represents the (abs_diff/resolution)</center>



* Of course, we can also compare the two DLS data:

<img src = "/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/chd_dls_dls.png" align="center" style="zoom:100%">

<center>Fig-10. two different DLS canopy height difference, resolution=0.1m, the colorbar represents the (abs_diff/resolution) </center>

* Overall, we can see that the registration between DLS data is relatively good, and the difference in general local heights is zero. I'm afraid it won't work to compare the registration quality between DLS and TLS, the difference is too much.



# 3. Conclusion





# 4. Annexe

* slice_height=1m, voxel_size=0.05m

  * coincident rate

    <img src = "/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/coi_rate_sh_1m_vs_005m_gs_18m.png" align="center" style="zoom:60%">

  * voxel number

    <img src = "/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/nb_voxel_sh_1m_vs_005m_gs_18m.png" align="center" style="zoom:60%">

* slice_height=1m, voxel_size=0.2m

  * coincident rate

    <img src = "/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/coi_rate_sh_1m_vs_02m_gs_18m.png" align="center" style="zoom:60%">

  * voxel number

    <img src = "/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/nb_voxel_sh_1m_vs_02m_gs_18m.png" align="center" style="zoom:60%">

* slice_height=2m, voxel_size=0.05m

  * coincident rate

    <img src = "/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/coi_rate_sh_2m_vs_005m_gs_18m.png" align="center" style="zoom:60%">

  * voxel number

    <img src = "/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/nb_voxel_sh_2m_vs_005m_gs_18m.png" align="center" style="zoom:60%">

* slice_height=2m, voxel_size=0.1m

  * coincident rate

    <img src = "/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/coi_rate_sh_2m_vs_01m_gs_18m.png" align="center" style="zoom:60%">

  * voxel number

    <img src = "/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/nb_voxel_sh_2m_vs_01m_gs_18m.png" align="center" style="zoom:60%">

* slice_height=2m, voxel_size=0.2m

  * coincident rate

    <img src = "/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/coi_rate_sh_2m_vs_02m_gs_18m.png" align="center" style="zoom:60%">

  * voxel number

    <img src = "/home/yuchen/Documents/PhD/phd_mission/scripts/Calage/plot_rate_nb_voxel/nb_voxel_sh_2m_vs_02m_gs_18m.png" align="center" style="zoom:60%">
