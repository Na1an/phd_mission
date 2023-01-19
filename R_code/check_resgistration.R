library("lidR")
library("terra")
library("raster")

##############################
# Script created on 22-05-16 #
##############################

########################
# Step-1. prepare data #
########################
dls1 <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-05-16_vol_trajet_new_vol_data_after_icp/YS-20211018-195956_ICPMeshCor_grd_clp-1ha-buf10m.laz")
dls2 <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-05-16_vol_trajet_new_vol_data_after_icp/YS-20211018-204147_grd_clp-1ha-buf10m.laz")

plot(dls1, main="dls vol 1", backend="lidRviewer")
plot(dls2, main="dls vol 2", backend="lidRviewer")

# step classify noise, noise point will be classified as 18
dls1 <- classify_noise(dls1, sor(10,10))
plot(dls1, main="dls denoise vol 1", backend="lidRviewer")
table(dls1@data$Classification)
dls1 <- dls1[dls1@data$Classification != 18]
table(dls1@data$Classification)

dls2 <- classify_noise(dls2, sor(10,10))
plot(dls2, main="dls denoise vol 2", backend="lidRviewer")
table(dls2@data$Classification)
dls2 <- dls2[dls2@data$Classification != 18]
table(dls2@data$Classification)

# prepare merged data - dls vol 1+2
dls12 <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-05-16_vol_trajet_new_vol_data_after_icp/two_vol_merged.las")
dls12 <- classify_noise(dls12, sor(10,10))
dls12 <- dls12[dls12@data$Classification != 18]
table(dls12@data$Classification)

# prepare TLS data decimated
tls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/21-12-08_tls_merged_moins_dense_Paracou/GLCS_merged_mon.laz")
tls <- classify_noise(tls, sor(10,10))
table(tls@data$Classification)
tls <- tls[tls@data$Classification != 18]
table(tls@data$Classification)

#######################
# Step-2. compare DTM #
#######################

# dls1 dtm
dls1_dtm <- grid_terrain(dls1,res = 0.5, use_class = c(2), algorithm = tin())
plot(dls1_dtm, main="dls vol 1 - [195956], dtm")

# dls2 dtm
dls2_dtm <- grid_terrain(dls2,res = 0.5, use_class = c(2), algorithm = tin())
plot(dls2_dtm, main="dls vol 2 - [204147], dtm")

# dls12 dtm
dls12_dtm <- grid_terrain(dls12,res = 0.5, use_class = c(2), algorithm = tin())
plot(dls12_dtm, main="dls vol 12 (merged), dtm")

# tls dtm
#tls_dtm <- grid_terrain(tls,res = 0.5, use_class = c(2), algorithm = tin())
mycsf <- csf(sloop_smooth = FALSE, class_threshold = 2, cloth_resolution = 15, time_step = 1)
tls <- classify_ground(tls, algorithm = csf())
tls_dtm <- grid_terrain(tls,res = 0.5, use_class = c(2), algorithm = tin())
plot(tls_dtm, main="tls ground point")

# really bad!!!!!!!!!!!!!!
# tooooooo bad result
plot(dls1_dtm - dls2_dtm, main="diff dtm : dls vol 1 - dls vol 2")
plot(dls1_dtm - tls_dtm, main="diff dtm : dls vol 1 - tls")
plot(dls2_dtm - tls_dtm, main="diff dtm : dls vol 1 - tls")
plot(dls12_dtm - tls_dtm, main="diff dtm : dls vol 12 - tls")

dls_dtm_dif <- dls1_dtm - dls2_dtm
dls_dtm_dif_reg <- focal(dls_dtm_dif, matrix(1,11,11), mean, na.rm=FALSE, pad=FALSE, padValue=NA, NAonly=FALSE)
plot(dls_dtm_dif_reg, main="dls_dtm_dif_reg")
#######################
# Step-3. compare CHM #
#######################
dls1_chm <- grid_canopy(dls1, res = 0.5, algorithm=p2r())
dls2_chm <- grid_canopy(dls2, res = 0.5, algorithm=p2r())
dls12_chm <- grid_canopy(dls12, res = 0.5, algorithm=p2r())

plot(dls1_chm, main="dls1_chm")
plot(dls2_chm, main="dls2_chm")
plot(dls12_chm, main="dls12 CHM")

tls_chm <- grid_canopy(tls, res = 0.5, algorithm=p2r())
plot(tls_chm, main="tls CHM")

dls_chm_dif <- dls1_chm - dls2_chm
dls_chm_dif_reg <- focal(dls_chm_dif, matrix(1,11,11), mean, na.rm=FALSE, pad=FALSE, padValue=NA, NAonly=FALSE)
plot(dls_chm_dif_reg, main="dls1_chm - dls2_chm, res=0.5m")

crs(dls1_chm) <- crs(tls_chm)
d1t_chm_dif <- dls1_chm - tls_chm
d1t_chm_dif_reg <- focal(d1t_chm_dif, matrix(1,11,11), mean, na.rm=FALSE, pad=FALSE, padValue=NA, NAonly=FALSE)
plot(d1t_chm_dif_reg, main="dls1_chm - tls_chm, res=0.5m")

plot(dls2_chm - tls_chm, main="dls2_chm - tls_chm, res=0.5m")
plot(dls12_chm - tls_chm, main="dls12_chm - tls_chm, res=0.5m")


dls1 <- readLAS("/home/yuchen/Documents/PhD/data_for_project/21-12-07_als_dls_DataCalibration_only_raw_data_no_seg_no_labeling_Nouragues/ALS_nougragues/ALS2018_AMAPvox.laz")
summary(dls1)

tls2 <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-05_tls_labelled_data_corrected/1_24/4.laz")
summary(tls2)

/home/yuchen/Documents/PhD/data_for_project/21-12-07_als_dls_DataCalibration_only_raw_data_no_seg_no_labeling_Nouragues/ALS_nougragues/ALS2018_AMAPvox.laz

