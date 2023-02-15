library("lidR")
library("dplyr")
library("ggplot2")

############## tls ##################

res_220 <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-05_tls_labelled_data_corrected/res_corrected_only_identified.las")

x <- res_220@data$WL
x <- replace(x, x == 1, 4)
res_220 <- add_lasattribute(res_220, x, "label", "Label for fine tuning")

#lidR::plot(res_220, color = "label", bg="white", legend=TRUE)

writeLAS(res_220, "tls_fine_tuning_res_all.las")

############# ULS #####################

uls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-09-01_new_data_with_new_feature/dls_new_test_with_new_feature.las")

x <- uls@data$WL
x <- replace(x, x == 1, 4)
uls <- add_lasattribute(uls, x, "label", "Label for fine tuning")

lidR::plot(uls, color = "label", bg="white", legend=TRUE)

writeLAS(uls, "uls_retrain_test.las")
