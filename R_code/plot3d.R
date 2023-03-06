library("lidR")
library("lattice")
library("ggplot2")
library("caret")
library("dplyr")
library("lidUrb")
library("plotly")
library("rgl")

lidR::plot(uls_all,color="WL",legend=TRUE)
uls_all <- readLAS("/home/yuchen/Documents/PhD/data_for_project/23-02-09_dls_remake_280_trees/output_file_test.las")
uls_all@data <- uls_all@data[uls_all@data$WL>1]
mycolors_bis <- c("chartreuse4","#91d024","#91d024") 
uls_all@data$colors <- mycolors[as.numeric(uls_all@data$WL)]
plot3d( 
  x=uls_all@data$X, 
  y=uls_all@data$linearity, 
  z=uls_all@data$sphericity, 
  col=uls_all@data$colors, 
  type = 'p', 
  xlab="X", ylab="linearity", zlab="sphericity")


uls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/23-02-09_dls_remake_280_trees/output_file_test.las")
uls@data <- uls@data[uls@data$WL<=2]
mycolors <- c("chartreuse4","#388498")
uls@data$WL <- uls@data$WL+1
uls@data$colors <- mycolors[as.numeric(uls@data$WL)]

table(uls@data$WL)

plot3d( 
  x=uls@data$Z, 
  y=uls@data$linearity, 
  z=uls@data$verticality, 
  col=uls@data$colors, 
  type = 'p', 
  xlab="Z", ylab="linearity", zlab="verticality")

lidR::plot(uls,color="WL",legend=TRUE)
plot3d( 
  x=uls@data$Y, 
  y=uls@data$linearity, 
  z=uls@data$sphericity, 
  col=uls@data$colors, 
  type = 'p', 
  xlab="Y", ylab="linearity", zlab="sphericity")

plot3d( 
  x=uls@data$Z, 
  y=uls@data$sphericity, 
  z=uls@data$verticality, 
  col=uls@data$colors, 
  type = 'p', 
  xlab="Z", ylab="sphericity", zlab="verticality")


plot3d( 
  x=uls@data$sphericity, 
  y=uls@data$linearity, 
  z=uls@data$verticality, 
  col=uls@data$colors, 
  type = 'p', 
  xlab="sphericity", ylab="linearity", zlab="verticality")


