library("lidR")
library("lattice")
library("ggplot2")
library("dplyr")

uls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-09-01_new_data_with_new_feature/dls_new_train_with_new_feature.las")
#summary(uls)
summary(uls)
uls@data$`Roughness (0.7)` <- (uls@data$Intensity/65535)*40 - 40

lidR::plot(uls,color="Intensity", legend=TRUE)
writeLAS(uls, "/home/yuchen/Documents/dls_reflectance.las")
