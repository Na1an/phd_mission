# store chm diff data to csv using dataframe
library(lattice)
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


############################################
# read .csv file and plot the distribution #
############################################
df_new <- read.csv("/home/yuchen/Desktop/res_25_04/res.csv")
hist(df_new$InterQuantile)
stripplot(df_new$Var)
stripplot(df_new$Var)
stripplot(df_new$Skewness)

show_plot <- function(x, title){
  h<-hist(x, breaks=10, col="white", main=title)
  xfit<-seq(min(x),max(x),length=24)
  yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
  yfit <- yfit*diff(h$mids[1:2])*length(x)
  lines(xfit, yfit, col="blue", lwd=2)
}
show_plot(df_new$Median, "CHM diff - Median")
show_plot(df_new$Mean, "CHM diff - Mean")
show_plot(df_new$Q1_4, "CHM diff - Q1_4")
show_plot(df_new$Q2_4, "CHM diff - Q2_4")
show_plot(df_new$Q3_4, "CHM diff - Q3_4")
show_plot(df_new$InterQuantile, "CHM diff - InterQuantile")
show_plot(df_new$Skewness, "CHM diff - Skewness")
show_plot(df_new$Kurtosis, "CHM diff - Kurtosis")
show_plot(df_new$Sd, "CHM diff - Sd")
show_plot(df_new$Var, "CHM diff - Var")
show_plot(df_new$HarmonicMean, "CHM diff - HarmonicMean")
show_plot(df_new$VariationRatio, "CHM diff - VariationRatio")

