library("lidR")
library("dplyr")
library("lattice")
library("caret")
library("lidUrb")
library("ggplot2")

tls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-05_tls_labelled_data_corrected/res_corrected_only_identified_treeid_220.las")
predict <- readLAS("/home/yuchen/Documents/PhD/res_corrected_only_identified_treeid_220_FSCT_output_retrain/segmented.las")
summary(tls)
summary(predict)

tls@data <- tls@data[order(tls@data$X, tls@data$Y, tls@data$Z)]
predict@data <- predict@data[order(predict@data$X, predict@data$Y, predict@data$Z)]

head(tls@data[0:10])
head(predict@data[0:10])

# leave=0, wood=1
tls@data$WL <- replace(tls@data$WL, tls@data$WL==2, 0)
predict@data$label <- replace(predict@data$label, predict@data$label==1, 2)
predict@data$label <- replace(predict@data$label, predict@data$label==0, 1)
predict@data$label <- replace(predict@data$label, predict@data$label==2, 0)

truth <- factor(tls@data$WL)
pred <- factor(predict@data$label)

table(truth, pred)

draw_confusion_matrix <- function(cm, f_name) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title(paste('FSCT (retrain) on TLS - Confusion Matrix :',f_name), cex.main=2)
  
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
table(truth)
table(pred)
cm <- confusionMatrix(truth, pred)
draw_confusion_matrix(cm, "label")

p <- ggplot(data=tls@data, aes(x=WL)) + 
  geom_bar(aes(y = (after_stat(count)))) + 
  geom_bar(aes(x=WL),color="red")
p

p2 <- ggplot(data=predict@data, aes(x=label)) + 
  geom_bar(aes(y = (after_stat(count)))) + 
  geom_bar(aes(x=label),color="red")
p2
