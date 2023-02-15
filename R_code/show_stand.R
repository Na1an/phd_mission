library("lidR")

path <- "/home/yuchen/Documents/PhD/data_for_project/22-09-01_new_data_with_new_feature/dls_new_val_with_new_feature.las"
uls <- readLAS(path)

lidR::plot(uls,color="Roughness (0.7)", legend=TRUE)
d <- uls@data$`Roughness (0.7)`

uls@data$`Roughness (0.7)` <- (d - mean(d))/sd(d)
lidR::plot(uls,color="Roughness (0.7)", legend=TRUE)
d
