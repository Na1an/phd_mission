library("lidR")
library("terra")
library("raster")
library("sf")
library("mapview")
library("moments")
# crop dls data to 1ha region
roi_info <- st_read("/home/yuchen/Documents/PhD/data_for_project/22-04-28_dtm_donnee_vol/PLotCNES_oneha.gpkg")
dls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-01-11_dls_2_vol_same_places_Paracou/YS-20211018-204147.laz")
#crs(dls) <- crs(roi_info)
tls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-05_tls_labelled_data_corrected/1.laz")
res = 0.05
dls_cp <- rasterize_canopy(dls, res=res, algorithm = p2r())
tls_cp <- rasterize_canopy(tls, res=res, algorithm = p2r())
plot(crop(dls_cp,tls_cp))

dls_roi <- clip_roi(dls, roi_info)
plot(dls_roi)

# compare cropped data
dls <- readLAS("/home/yuchen/Documents/PhD/phd_mission/scripts/Crop_data/dls_data_croped_1_2022-04-27_16-32-23.las")
tls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-05_tls_labelled_data_corrected/1_24/1.laz")
res=0.1

dls_top <- rasterize_canopy(dls, res=res, algorithm = p2r())
tls_top <- rasterize_canopy(tls, res=res, algorithm = p2r())
plot(dls_top)
plot(tls_top)
crs(dls_top) <- crs(tls_top)
dif <- dls_top - tls_top
dif_reg <- focal(dif, matrix(1,7,7), mean, na.rm=FALSE, pad=FALSE, padValue=NA, NAonly=FALSE)
plot(dif_reg)

# !!!!!!!!!!!!!!!!!!! todo:
# modify script R, change it to project/create roi directly inside the R code
crop(dls,tls)


#######################
# start 1-24 percells #
#######################
dls <- readLAS("/home/yuchen/Documents/PhD/phd_mission/scripts/Crop_data/dls_data_croped2022-06-12_12-05-19.las")
tls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-05_tls_labelled_data_corrected/1_24/6.laz")
res = 0.5
dls_cp <- rasterize_canopy(dls, res=res, algorithm = p2r())
tls_cp <- rasterize_canopy(tls, res=res, algorithm = p2r())
plot(dls_cp)
plot(tls_cp)
crs(dls_cp) <- crs(tls_cp)
dif_025 <- crop(dls_cp, tls_cp) - tls_cp
dif_reg <- focal(dif, matrix(1,7,7), mean, na.rm=FALSE, pad=FALSE, padValue=NA, NAonly=FALSE)
plot(dif_025)

col <- heat.colors(3)
plot(dls_cp, col=col)
plot(tls_cp)

res = 0.5
dls_cp <- rasterize_canopy(dls, res=res, algorithm = p2r())
tls_cp <- rasterize_canopy(tls, res=res, algorithm = p2r())
crs(dls_cp) <- crs(tls_cp)
dif_050 <- crop(dls_cp, tls_cp) - tls_cp
#dif_reg <- focal(dif, matrix(1,7,7), mean, na.rm=FALSE, pad=FALSE, padValue=NA, NAonly=FALSE)
plot(dif_050, main="chm_dif res=0.5m", cex.main=1.5)

res = 1.0
dls_cp <- rasterize_canopy(dls, res=res, algorithm = p2r())
tls_cp <- rasterize_canopy(tls, res=res, algorithm = p2r())
crs(dls_cp) <- crs(tls_cp)
dif_100 <- crop(dls_cp, tls_cp) - tls_cp
#dif_reg <- focal(dif, matrix(1,7,7), mean, na.rm=FALSE, pad=FALSE, padValue=NA, NAonly=FALSE)
plot(dif_100, main="chm_dif res=1m", cex.main=1.5)
# 1.laz
#tls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-05_tls_labelled_data_corrected/1.laz")




tls_01 <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-05_tls_labelled_data_corrected/1.laz")
plot(chm_comp(dls_cp, tls_01, res))

tls_02 <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-05_tls_labelled_data_corrected/2.laz")
plot(chm_comp(dls_cp, tls_02, res))

tls_tmp <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-05_tls_labelled_data_corrected/3.laz")
plot(chm_comp(dls_cp, tls_tmp, res))


# chm : canopy height model
chm_comp <- function(dls_cp, tls_raw, res){
  tls_cp <- rasterize_canopy(tls_raw, res=res, algorithm = p2r())
  crs(tls_cp) <- crs(dls_cp)
  dif <- crop(dls_cp, tls_cp) - tls_cp
  #dif_reg <- focal(dif, matrix(1,3,3), mean, na.rm=FALSE, pad=FALSE, padValue=NA, NAonly=FALSE)
  dif
}
dls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-01-11_dls_2_vol_same_places_Paracou/YS-20211018-204147.laz")
res = 0.5
dls_cp <- rasterize_canopy(dls, res=res, algorithm = p2r())
out_path <- "/home/yuchen/Desktop/res_25_04/chm_compare"
int_path <- "/home/yuchen/Documents/PhD/data_for_project/22-04-05_tls_labelled_data_corrected/"
for (i in 1:24){
  print(paste("> start",toString(i)))
  i_f <- formatC(i, width=3, flag="0")
  out <- paste("/home/yuchen/Desktop/res_25_04/chm_compare_", toString(i_f),".las", sep="")
  inp <- paste("/home/yuchen/Documents/PhD/data_for_project/22-04-05_tls_labelled_data_corrected/", toString(i),".laz", sep="")
  tls_tmp <- readLAS(inp)
  #png(out)
  exit()
  plot(chm_comp(dls_cp, tls_tmp, res))
  dev.off()
  print(paste("> ok",toString(i)))
  
}


################
# catalog .las #
################
# load a all files (.las) in a directory
ctg <- readLAScatalog("/home/yuchen/Documents/PhD/data_for_project/22-01-11_dls_2_vol_same_places_Paracou/YS-20211018-204147.laz")
plot(ctg, map=TRUE)

ctg <- readLAScatalog("/home/yuchen/Documents/PhD/data_for_project/22-04-05_tls_labelled_data_corrected/1_24")
chm <- rasterize_canopy(ctg, 1.0, p2r())
col <- random.colors(50)
plot(chm, col=col)


#########################
# distribution included #
#########################
# chm : canopy height model
chm_comp <- function(dls_cp, tls_raw, res){
  tls_cp <- rasterize_canopy(tls_raw, res=res, algorithm = p2r())
  crs(tls_cp) <- crs(dls_cp)
  dif <- crop(dls_cp, tls_cp) - tls_cp
  #dif_reg <- focal(dif, matrix(1,3,3), mean, na.rm=FALSE, pad=FALSE, padValue=NA, NAonly=FALSE)
  dif
}

dls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-01-11_dls_2_vol_same_places_Paracou/YS-20211018-204147.laz")
res = 0.5
dls_cp <- rasterize_canopy(dls, res=res, algorithm = p2r())
out_path <- "/home/yuchen/Desktop/res_25_04/chm_compare"
int_path <- "/home/yuchen/Documents/PhD/data_for_project/22-04-05_tls_labelled_data_corrected/"

Id <- c()
Q1_4 <- c()
Q2_4 <- c()
Q3_4 <- c()
InterQuantile <- c()
Mean <- c()
Median <- c()
Skewness <- c()
Kurtosis <- c()
Sd <- c()
Var <- c()
HarmonicMean <- c()
VariationRatio <- c()

for (i in 1:24){
  print(paste("> start",toString(i)))
  i_f <- formatC(i, width=3, flag="0")
  out <- paste("/home/yuchen/Desktop/res_25_04/chm_compare_", toString(i_f),".las", sep="")
  inp <- paste("/home/yuchen/Documents/PhD/data_for_project/22-04-05_tls_labelled_data_corrected/1_24/", toString(i),".laz", sep="")
  tls_tmp <- readLAS(inp)
  dif <- chm_comp(dls_cp, tls_tmp, res) 
  #png(out)
  data <- as.vector(dif$Z)
  #plot(dif)
  Id <- append(Id, i, after=length(Id))
  Q1_4 <- append(Q1_4, as.numeric(quantile(data, 0.25)), after=length(Q1_4))
  Q2_4 <- append(Q2_4, as.numeric(quantile(data, 0.5)), after=length(Q2_4))
  Q3_4 <- append(Q3_4, as.numeric(quantile(data, 0.75)), after=length(Q3_4))
  InterQuantile <- append(InterQuantile, as.numeric(quantile(data, 0.75)) - as.numeric(quantile(data, 0.25)), after=length(InterQuantile))
  Mean <- append(Mean, mean(data), after=length(Mean))
  Median <- append(Median, median(data), after=length(Median))
  Skewness <- append(Skewness, skewness(data), after=length(Skewness))
  Kurtosis <- append(Kurtosis, kurtosis(data), after=length(Kurtosis))
  Sd <- append(Sd, sd(data), after=length(Sd))
  Var <- append(Var, var(data), after=length(Var))
  HarmonicMean <- append(HarmonicMean, 1/mean(1/data), after=length(HarmonicMean))
  VariationRatio <- append(VariationRatio, 1 - max(data)/sum(data), after=length(VariationRatio) )

  #dev.off()
  print(paste("> ok",toString(i)))
  
}
df <-NA
df <- data.frame(
  Id, Q1_4, Q2_4, Q3_4, InterQuantile, Mean, Median, Skewness, Kurtosis, Sd, Var, HarmonicMean, VariationRatio
)
print(df)
write.csv(df, "/home/yuchen/Desktop/res_25_04/res.csv")
