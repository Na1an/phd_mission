library("lidR")

for (i in 1:169){
  print(paste("> start ",toString(i)))
  index_f <- formatC(i, width=3, flag="0")
  data_raw <- paste("/home/yuchen/Documents/PhD/data_for_project/22-07-04_new_data/Plot_100_100_raw_Segmented/", toString(i),".laz", sep="")
  data_filtered <- paste("/home/yuchen/Documents/PhD/data_for_project/22-07-04_new_data/Plot_100_100_raw_Segmented/filtered/", toString(i_f),".las", sep="")
  data_filtered <- data_filtered$WL
  
  print(paste("< end ",toString(i)))
}