library("lidR")
library("lattice")
library("ggplot2")
library("caret")
library("dplyr")
library("lidUrb")
library("plotly")
library("rgl")

########################################################
# functions
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# be attention!
# here, is we use the function confusionMatrix of caret
# the label=1 is the specificity

draw_confusion_matrix <- function(cm, f_name) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title(paste('Confusion Matrix :',f_name), cex.main=2)
  
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

pred <- readLAS("/home/yuchen/Documents/PhD/data_for_project/23-02-28_predic_result/res_remove_duplicated_points.las")
pred@data <- pred@data[pred@data$true >=0 ]

lidR::plot(pred, color = "true", size=2, colorPalette = c("#93c555","#007f54"), bg="white", legend=TRUE)
pred@data$true <- replace(pred@data$true, pred@data$true==1, 2)
pred@data$true <- replace(pred@data$true, pred@data$true==0, 1)
pred@data$true <- replace(pred@data$true, pred@data$true==2, 0)

pred@data$predict <- replace(pred@data$predict, pred@data$predict==1, 2)
pred@data$predict <- replace(pred@data$predict, pred@data$predict==0, 1)
pred@data$predict <- replace(pred@data$predict, pred@data$predict==2, 0)

hist(pred@data$wood_proba)
pred@data$wood_proba <- replace(pred@data$wood_proba, pred@data$wood_proba==-1, 0)
lidR::plot(pred, color = "wood_proba", size=2, colorPalette = c("#93c555","#007f54"), bg="white", legend=TRUE)

hist(-log(pred@data$wood_proba))
pred@data$wood_proba_new <- (-log(pred@data$wood_proba))

hist(pred@data$wood_proba)
hist(-log(pred@data$wood_proba), breaks = 100)

pred@data[,new_predict := as.numeric(wood_proba_new < 6)]

hist(pred@data$wood_proba_new)
hist(pred@data$new_predict)

truth <- factor(pred@data$true)
predict <- factor(pred@data$new_predict)

hist(pred@data$true)
hist(pred@data$new_predict)
table(truth)
table(predict)

cm <- confusionMatrix(predict, truth)
cm
draw_confusion_matrix(cm, "Yuchen model")

