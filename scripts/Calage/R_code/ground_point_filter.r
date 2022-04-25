library("lidR")
#tls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-15_calage/dls_data_calage.las")
tls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-05_tls_labelled_data_corrected/2.laz")
#print(tls)
#summary(tls)
las_check(tls)
#plot(tls)
plot(tls, color="Intensity", breaks="quantile", bg="white", axis=TRUE, legend=TRUE)

# vox
#vox <- voxelize_points(tls, 1)
#plot(vox, voxel=TRUE, bg="white")
dls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-01-11_dls_2_vol_same_places_Paracou/vol_195956.las")
mycsf <- csf(sloop_smooth = TRUE, class_threshold = 1, cloth_resolution = 1, time_step = 1)
dls <- classify_ground(dls, mycsf)
gnd <- filter_ground(dls)
plot(gnd, bg="white", axis=TRUE, legend=TRUE)
writeLAS(gnd, "/home/yuchen/Documents/PhD/data_for_project/22-01-11_dls_2_vol_same_places_Paracou/dls_terrain_lidR.las")

tls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-22_lidr_data/res_2345.las")
mycsf <- csf(sloop_smooth = TRUE, class_threshold = 1, cloth_resolution = 1, time_step = 1)
tls <- classify_ground(tls, mycsf)
gnd <- filter_ground(tls)
plot(gnd, bg="white", axis=TRUE, legend=TRUE)
writeLAS(gnd, "/home/yuchen/Documents/PhD/data_for_project/22-04-22_lidr_data/tls_terrain_lidR.las")

##############
# start our recipe
##############



