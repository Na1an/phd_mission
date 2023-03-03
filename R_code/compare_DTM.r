library(lidR)
library(terra) #modern version of raster to be preferred!
library(sf)

#plot CNES with buffer

plot_1ha_no_buf<-st_read("d:/Mes Donnees/GVincent/DataGuyane/PARACOU/TLS_CNES_Biomasse/ParcelleBiomasse_SHP", "PlotCNES")
plot_1ha_10mbuf<-st_read("d:/Mes Donnees/GVincent/DataGuyane/PARACOU/TLS_CNES_Biomasse/ParcelleBiomasse_SHP", "CNES_oneha_10mBuff")

plot(x=vect(plot_1ha_10mbuf), col="blue")

setwd("d:/Mes Donnees/GVincent/DataGuyane/PARACOU/DLS_CNES_Biomasse")
las1=readLAS("YS-20211018-195956/export/YS-20211018-195956_ICPMeshCor_grd.laz")
st_crs(las1)<-st_crs(plot_1ha_no_buf)
cnes1<-clip_roi(las1,plot_1ha_no_buf)
plot(cnes1)
cnes1_buf<-clip_roi(las1,plot_1ha_10mbuf)
plot(cnes1_buf)
rm(las1)
gc()
## denoise DLS first (BEWARE deletes too many ground points, OK for above canopy outliers only)
cnes1_buf_dnz<-classify_noise(cnes1_buf, sor(10,10))
table(cnes1_buf_dnz@data$Classification)

#HERE WE USE DATA WITHOUT ICP CORRECTION - BETTER!!
las3=readLAS("YS-20211018-204147/export/YS-20211018-204147_grd.laz")
st_crs(las3)<-st_crs(plot_1ha_10mbuf)

cnes3_buf<-clip_roi(las3,plot_1ha_10mbuf)
#denoise
cnes3_buf_dnz<-classify_noise(cnes3_buf, sor(10,10))
table(cnes3_buf_dnz@data$Classification)

#writeLAS(cnes1_buf,"YS-20211018-195956/export/YS-20211018-195956_ICPMeshCor_grd_clp-1ha-buf10m.laz")
#writeLAS(cnes3_buf,"YS-20211018-204147/export/YS-20211018-204147_grd_clp-1ha-buf10m.laz")

cnes1_dtm<-grid_terrain(cnes1_buf,res = 0.5, use_class = c(2), algorithm = tin())
cnes3_dtm<-grid_terrain(cnes3_buf,res = 0.5, use_class = c(2), algorithm = tin())

cnes1_csm<-grid_canopy(cnes1_buf_dnz, res = 0.5, algorithm=p2r())
cnes3_csm<-grid_canopy(cnes3_buf_dnz, res = 0.5, algorithm=p2r())

plot(cnes1_csm-cnes3_csm)
plot(cnes1_dtm-cnes3_dtm)

TLS=readLAS("e:/TLS-2021-CNES/LazFormat/TLS_UTM_merged_grd.laz")
st_crs(TLS)<-st_crs(plot_1ha_10mbuf)
TLS<-clip_roi(TLS, plot_1ha_10mbuf)
plot(TLS)

## consider denoising TLS as well??
TLS_dtm<-grid_terrain(TLS,res = 0.5, use_class = c(2), algorithm = tin())
TLS_csm<-grid_canopy(TLS, res = 0.5, algorithm=p2r())

plot(cnes1_dtm-cnes3_dtm)
plot(cnes1_dtm-TLS_dtm)
plot(cnes3_dtm-TLS_dtm)

dif_dtm=cnes1_dtm-cnes3_dtm
dif_dtm_reg=focal(dif_dtm, matrix(1,11,11), mean, na.rm=FALSE, pad=FALSE, padValue=NA, NAonly=FALSE)
plot(dif_dtm)
plot(dif_dtm_reg, main="differences between DLS ground models")

dif_csm=cnes1_csm-cnes3_csm
dif_csm_reg=focal(dif_csm, matrix(1,11,11), mean, na.rm=T, pad=FALSE, padValue=NA, NAonly=FALSE)
mean(dif_csm[], na.rm=T)
sd(dif_csm[], na.rm=T)
mean(dif_csm_reg[], na.rm=T)
sd(dif_csm_reg[], na.rm=T)

plot(dif_csm, main="df DLS")
plot(dif_csm_reg, main="differences between DLS canopy models")

dif_dtm_tls=cnes1_dtm-TLS_dtm
dif_dtm_tls3=cnes3_dtm-TLS_dtm
dif_dtm_tls_reg=focal(dif_dtm_tls, matrix(1,11,11), mean, na.rm=FALSE, pad=FALSE, padValue=NA, NAonly=FALSE)
dif_dtm_tls_reg3=focal(dif_dtm_tls3, matrix(1,11,11), mean, na.rm=FALSE, pad=FALSE, padValue=NA, NAonly=FALSE)
plot(dif_dtm_tls)
plot(dif_dtm_tls_reg, main="differences between TLS and DLS1 ground models")
plot(dif_dtm_tls_reg3, main="differences between TLS and DLS2 ground models")
mean(dif_dtm_tls[], na.rm=T)
sd(dif_dtm_tls[], na.rm=T)
mean(dif_dtm_tls3[], na.rm=T)
sd(dif_dtm_tls3[], na.rm=T)

dif_csm_tls=cnes1_csm-TLS_csm
dif_csm_tls_reg=focal(dif_csm_tls, matrix(1,11,11), mean, na.rm=FALSE, pad=FALSE, padValue=NA, NAonly=FALSE)
plot(dif_csm_tls)
plot(dif_csm_tls_reg, main="differences between TLS and DLS1 canopy models")

dif_csm3_tls=cnes3_csm-TLS_csm
dif_csm3_tls_reg=focal(dif_csm3_tls, matrix(1,11,11), mean, na.rm=FALSE, pad=FALSE, padValue=NA, NAonly=FALSE)
plot(dif_csm3_tls)
plot(dif_csm3_tls_reg, main="differences between TLS and DLS2 canopy models")
