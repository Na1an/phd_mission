library("lidR")
library("lattice")
library("ggplot2")
library("caret")
library("dplyr")
library("lidUrb")

# functions
draw_confusion_matrix <- function(cm, f_name) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title(paste('Only use linearity on ULS - Confusion Matrix :',f_name), cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'Leaf', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'Wood', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'True Label', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'Leaf', cex=1.2, srt=90)
  text(140, 335, 'Wood', cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}


uls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-11_transposed_data/dls_only_sutdy_region.las")
#summary(uls)
summary(uls)
uls@data <- uls@data[uls@data[["llabel"]]>1]
lidR::plot(uls, color = "llabel", size=2, colorPalette = c("#93c555","#007f54"), bg="white", legend=TRUE)

uls@data$llabel <- replace(uls@data$llabel, uls@data$llabel == 2, 1)
uls@data$llabel <- replace(uls@data$llabel, uls@data$llabel == 3, 0)


#lidR::plot(uls,color="llabel",size=2,colorPalette = c("chartreuse4","cornsilk2"), legend=TRUE)
#p <- ggplot(data=uls@data, aes(x=llabel)) + geom_bar(aes(y = (..count..)))
#p

uls_lewos <- LW_segmentation_graph(uls)

# density distribution of p_wood
p2 <- ggplot(uls_lewos@data, aes(x=p_wood)) + geom_density()
p2

# density distribution of SoD
p <- ggplot(uls_lewos@data, aes(x=SoD)) + geom_density()
p

uls_lewos@data[,wood_p := as.numeric(p_wood >= 0.9)]
uls_lewos@data[,wood_s := as.numeric(SoD >= 0.99)]
lidR::plot(uls_lewos,color="wood_s",size=2,colorPalette = c("chartreuse4","cornsilk2"), legend=TRUE)
lidR::plot(uls_lewos,color="wood_p",size=2,colorPalette = c("chartreuse4","cornsilk2"), legend=TRUE)
lidR::plot(uls_lewos,color="llabel",size=2,colorPalette = c("chartreuse4","cornsilk2"), legend=TRUE)

#lidR::plot(uls_lewos, color="llabel",size=2,colorPalette = c("chartreuse4","cornsilk2"), legend=TRUE)

truth <- factor(uls_lewos@data$llabel)
pred <- factor(uls_lewos@data$wood_p)
pred2 <- factor(uls_lewos@data$wood_s)

table(truth, pred)
table(truth, pred2)

cm <- confusionMatrix(pred, truth)
draw_confusion_matrix(cm, "wood_p")

cm2 <- confusionMatrix(pred2, truth)
draw_confusion_matrix(cm2, "wood_s")


##################### calculate confusion matrix #################

#uls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-09-01_new_data_with_new_feature/dls_new_test_with_new_feature.las")
uls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/23-02-28_predic_result/res_on_test_data_yuchen_model_remove_duplicated_points.las")
#uls_pred <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-09-01_new_data_with_new_feature/dls_new_test_with_new_feature_FSCT_output/segmented.las")
uls_pred <- readLAS("/home/yuchen/Documents/PhD/data_for_project/23-02-28_predic_result/res_on_test_data_yuchen_model_remove_duplicated_points.las")

true_data <- uls@data
pred_data <- uls_pred@data

table(true_data$true)
table(pred_data$predict)

true_data$WL <- replace(true_data$WL, true_data$WL == 2, 0)
pred_data$label <- replace(pred_data$label, pred_data$label == 1, 0)
pred_data$label <- replace(pred_data$label, pred_data$label == 3, 1)

table(true_data$WL)
table(pred_data$label)

true_data$true <- replace(true_data$true, true_data$true == 0, -1)
true_data$true <- replace(true_data$true, true_data$true == 1, 0)
true_data$true <- replace(true_data$true, true_data$true == -1, 1)

true_data$predict <- replace(true_data$predict, true_data$predict == 0, -1)
true_data$predict <- replace(true_data$predict, true_data$predict == 1, 0)
true_data$predict <- replace(true_data$predict, true_data$predict == -1, 1)

truth <- factor(true_data$WL)
pred <- factor(true_data$label)

truth <- factor(true_data$true)
pred <- factor(true_data$predict)

cm <- confusionMatrix(pred, truth)
draw_confusion_matrix(cm, "downgraded SOUL model")
