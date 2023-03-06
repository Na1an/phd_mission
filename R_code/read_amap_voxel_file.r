library("AMAPVox")
library("ggplot2")

vx <- readVoxelSpace("/home/yuchen/Documents/PhD/data_for_project/22-05-16_vol_trajet_new_vol_data_after_icp/uls_05.vox")

l_h <- 1
voxel_size <- 0.5
step <- (l_h/voxel_size)
h_min <- min(vx@data$k)
h_max <- max(vx@data$k)

occ_rate <- c()
intensity <- c()
sampling <- c()
height <- c()
# occlusion ratio
for (h in seq(h_min, h_max, by=step)){
  cat("h=", h, '\n')
  data_tmp <- vx@data[vx@data$k >= h & vx@data$k < h+step]
  intensity <- c(intensity, mean(data_tmp$lMeanTotal))
  sampling <- c(sampling, mean(data_tmp$nbSampling))
  
  data_tmp <- data_tmp[data_tmp$bsPotential != 0]
  cat("mean occ rate=", mean(data_tmp$bsEntering/data_tmp$bsPotential), '\n')
  occ_rate <- c(occ_rate, 1 - mean(data_tmp$bsEntering/data_tmp$bsPotential))
  height <- c(height, h*voxel_size)
}

df <- data.frame(Height=height, OcclusionRate=occ_rate, Intensity=intensity, Sampling=sampling)
df

hist(vx$nbSampling)

ci_occ <- (sd(occ_rate) * 1.96)/(length(occ_rate)**0.5)
ci_int <- (sd(intensity) * 1.96)/(length(intensity)**0.5)
ci_smp <- (sd(sampling) * 1.96)/(length(sampling)**0.5)

cat(ci_occ,'\t',ci_int,'\t',ci_smp, '\n')

## plot ###
p1<- ggplot(df, aes(x = Height, y = OcclusionRate)) +
  geom_ribbon(aes(ymin =OcclusionRate - ci_occ, ymax = OcclusionRate + ci_occ), 
              alpha = .3, fill = "darkseagreen3", color = "transparent") +
  geom_line(color = "aquamarine4",  lwd=.7) +
  labs(x = "Heights", y = "Occlusion Rate (95% CI)") +
  coord_flip() + 
  theme(text = element_text(size=15), axis.text=element_text(size=12))
p1


## intensity ###
p2<- ggplot(df, aes(x = Height, y = Intensity)) +
  geom_ribbon(aes(ymin = Intensity - ci_int, ymax = Intensity + ci_int), 
              alpha = .3, fill = "darkseagreen3", color = "transparent") +
  geom_line(color = "aquamarine4",  lwd=.7) +
  labs(x = "Heights", y = "lMeanTotal (95% CI)") +
  coord_flip() + 
  theme(text = element_text(size=15), axis.text=element_text(size=12))
p2

## sampling ###
p3<- ggplot(df, aes(x = Height, y = Sampling)) +
  geom_ribbon(aes(ymin = Sampling - ci_smp, ymax = Sampling + ci_smp), 
              alpha = .3, fill = "darkseagreen3", color = "transparent") +
  geom_line(color = "aquamarine4",  lwd=.7) +
  labs(x = "Heights", y = "Sampling (95% CI)") +
  coord_flip() + 
  theme(text = element_text(size=15), axis.text=element_text(size=12))
p3

