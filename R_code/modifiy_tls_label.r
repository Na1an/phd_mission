library("lidR")

tls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-05_tls_labelled_data_corrected/res_corrected_only_identified_treeid_200.las")

tls@data$WL <- replace(tls@data$WL, tls@data$WL==1, 4)
writeLAS(tls, "tls_fine_tuning_res_200.las")

path <- "/home/yuchen/Documents/PhD/data_for_project/22-07-04_new_data/Plot_100_100_raw_Segmented/dls_tranposed/dls_merged_230209.las"
uls <- readLAS(path)
length(unique(uls@data$treeID))
