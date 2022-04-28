library("lidR")
library("ggplot2")
library("terra")
library("raster")
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
dls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-22_lidr_data/dls_data_croped2022-04-25_14-35-48.las")
#mycsf <- csf(sloop_smooth = TRUE, class_threshold = 2, cloth_resolution = 15, time_step = 1)
#dls <- classify_ground(dls, mycsf)
dls <- classify_ground(dls, algorithm = pmf(ws = 5, th = 0.8))
gnd <- filter_ground(dls)
plot(gnd, bg="white", axis=TRUE, legend=TRUE)
writeLAS(gnd, "/home/yuchen/Desktop/res_25_04/dls_dtm_pmf_5_08.las")

#tls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-22_lidr_data/res_2345.las")
tls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-05_tls_labelled_data_corrected/3.laz")
mycsf <- csf(sloop_smooth = TRUE, class_threshold = 0.8, cloth_resolution = 0.1, time_step = 1)
tls <- classify_ground(tls, mycsf)
#tls <- classify_ground(tls, algorithm = pmf(ws = 5, th = 3))
gnd <- filter_ground(tls)
plot(gnd, bg="white", axis=TRUE, legend=TRUE)
writeLAS(gnd, "/home/yuchen/Desktop/res_25_04/tls_dtm_08_01.las")

##############
# start our recipe
##############


##############################
# cross section and triangle #
##############################

# function cross section
plot_crossection <- function(las,
                             p1 = c(min(las@data$X), mean(las@data$Y)),
                             p2 = c(max(las@data$X), mean(las@data$Y)),
                             width = 4, colour_by = NULL)
{
  colour_by <- enquo(colour_by)
  data_clip <- clip_transect(las, p1, p2, width)
  p <- ggplot(data_clip@data, aes(X,Z)) + geom_point(size = 0.5) + coord_equal() + theme_minimal()
  
  if (!is.null(colour_by))
    p <- p + aes(color = !!colour_by) + labs(color = "")
  
  return(p)
}

ggplot(dtm@data, aes(X,Z, color = Z)) + 
  geom_point(size = 0.2) + 
  coord_equal() + 
  theme_minimal() +
  scale_color_gradientn(colours = height.colors(50))

dtm <- readLAS("/home/yuchen/Desktop/res_25_04/tls_dtm_055_06.las")
#dtm <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-22_lidr_data/dls_data_croped2022-04-25_15-13-46.las")
#dtm <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-22_lidr_data/dls_data_croped2022-04-25_15-13-46.las")
#dtm$Classification <- 2L
#plot_crossection(dtm, colour_by = factor(Classification))

dtm_tin <- rasterize_terrain(dtm, res = 0.5, algorithm = tin())
plot_dtm3d(dtm_tin, bg = "white")
dtm_tin$Z[22.5, 22.5]
'''
nrow(dtm_tin)
ncol(dtm_tin)
dtm_tin[1,]
dtm_tin[2,]
dtm_tin[22,]
typeof(dtm_tin$Z$Z)
'''
writeLAS(dtm_tin, "/home/yuchen/Desktop/res_25_04/raster_triangle/dtm_raster_triangle.las")

###########
# restart #
###########
#dtm <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-22_lidr_data/dls_data_croped2022-04-25_15-13-46.las")
dtm <- readLAS("/home/yuchen/Desktop/res_25_04/tls_dtm_055_06.las")
#dtm$Classification <- 2L
dtm_tin <- rasterize_terrain(dtm, res = 0.5, algorithm = tin())
dim(dtm_tin)
dim(matrix(dtm_tin, nrow=20, byrow=TRUE))
matrix(dtm_tin, nrow=20, byrow=TRUE)
plot_dtm3d(dtm_tin, bg = "white")

dtm_tls <- readLAS("")

# comparaison des rasters de delta Z entre modèles de sol

comp_dtm_raster_z <- function(dls_path, tls_path, out_path, res=0.5, show_raster=FALSE){
  dtm_dls <- readLAS(dls_path)
  dtm_dls$Classification <- 2L
  dtm_dls_tin <- rasterize_terrain(dtm_dls, res = res, algorithm = tin())
  print(dim(dtm_dls_tin))
  
  dtm_tls <- readLAS(tls_path)
  dtm_tls$Classification <- 2L
  dtm_tls_tin <- rasterize_terrain(dtm_tls, res = res, algorithm = tin())
  print(dim(dtm_tls_tin))
  
  if(show_raster){
    plot_dtm3d(dtm_dls_tin, bg = "white")
    plot_dtm3d(dtm_tls_tin, bg = "white")
  }
  #print(dim(matrix(dtm_dls_tin, nrow=20, byrow=TRUE, ncol = 20)))
  #print(dim(matrix(dtm_tls_tin, nrow=20, byrow=TRUE, ncol = 20)))
  #res <- matrix(dtm_dls_tin, nrow=20, byrow=TRUE, ncol = 20) - matrix(dtm_tls_tin, nrow=20, byrow=TRUE, ncol = 20)
  res <- dtm_dls_tin - dtm_tls_tin
  plot_dtm3d(res, bg = "white")
  writeRaster(res, out_path, overwrite=TRUE)
}

dls_path <- "/home/yuchen/Documents/PhD/data_for_project/22-04-22_lidr_data/dls_data_croped2022-04-25_23-46-10.las"
tls_path <- "/home/yuchen/Desktop/res_25_04/tls_dtm_055_06.las"
out_path <- "/home/yuchen/Desktop/res_25_04/diff_dtm_raster_id_3_bis.tif"
comp_dtm_raster_z(dls_path, tls_path, out_path, 0.5)

# comparaison des rasters de delta Z entre modèle de canopée
#tls_path <- "/home/yuchen/Documents/PhD/data_for_project/22-04-22_lidr_data/res_2345.las"
#dls_path <- "/home/yuchen/Desktop/res_25_04/canopy/dls_data_croped2022-04-26_01-10-18.las"
tls_path <- "/home/yuchen/Documents/PhD/data_for_project/22-01-11_dls_2_vol_same_places_Paracou/working_zone_bis.las"
dls_path <- "/home/yuchen/Documents/PhD/data_for_project/22-01-11_dls_2_vol_same_places_Paracou/working_zone_204147.las"
comp_canopy_raster_z <- function(tls_path, res=0.5){
  tls <- readLAS(tls_path)
  chm <- rasterize_canopy(tls, res=res, algorithm = p2r())
  col <- height.colors(25)
  plot(chm, col=col)
  chm
}

tls_chm <- comp_canopy_raster_z(tls_path)
dls_chm <- comp_canopy_raster_z(dls_path)

print(dim(tls_chm))
typeof(crs(tls_chm))

quest <- crs(tls_chm)
     
print(dim(dls_chm))
typeof(tls_chm)

typeof(tls_chm["Z"])
as.numeric(dls_chm$Z)


col <- height.colors(25)
dls_chm[is.na(dls_chm)] <- 0.0
tls_chm[is.na(tls_chm)] <- 0.0
print(dls_chm$Z)
plot(dls_chm$Z - tls_chm$Z, col=col)
typeof(diff_chm)

s <- global(diff_chm, sum)
s[1,1]/(40*40)

########################################
# 27/04/22
# crs() really important
####################"

tls_path <- "/home/yuchen/Documents/PhD/data_for_project/22-01-11_dls_2_vol_same_places_Paracou/working_zone_bis.las"
dls_path <- "/home/yuchen/Documents/PhD/data_for_project/22-01-11_dls_2_vol_same_places_Paracou/working_zone_204147.las"
#tls_path <- "/home/yuchen/Documents/PhD/data_for_project/22-04-22_lidr_data/res_2345.las"
#dls_path <- "/home/yuchen/Desktop/res_25_04/canopy/dls_data_croped2022-04-26_01-10-18.las"
comp_canopy_raster_z <- function(tls_path, res=1){
  tls <- readLAS(tls_path)
  chm <- rasterize_canopy(tls, res=res, algorithm = p2r())
  col <- height.colors(25)
  plot(chm, col=col)
  chm
}

tls_chm <- comp_canopy_raster_z(tls_path)
dls_chm <- comp_canopy_raster_z(dls_path)
crs(dls_chm) <- crs(tls_chm)
dif <- tls_chm - dls_chm
dif_reg <- focal(dif, matrix(1,7,7), mean, na.rm=FALSE, pad=FALSE, padValue=NA, NAonly=FALSE)
plot(dif_reg)
plot(tls_chm - dls_chm, col=height.colors(45))

