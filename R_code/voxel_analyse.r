library("VoxR")
library("lidR")
library("dplyr")
library("lattice")
library("caret")
library("lidUrb")
library("ggplot2")

# function
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
  
  # sensitivity
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  
  # specificity
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  
  # precision
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  
  # recall
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  
  # F1
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information
  # Accuracy
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  
  # Kappa
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
  
  return(list_rturn)
}

######################################################################
# Layer analyse, dose FSCT and lewos works also good on canopy part? #
######################################################################

# 1. load data
# res_corrected_only_identified_treeid_220 vs segmented_retrain_tls
tls <- readLAS("/home/yuchen/Documents/PhD/data_for_project/22-04-05_tls_labelled_data_corrected/res_corrected_only_identified_treeid_220.las")


# 2. fsct vs lewos_graph vs lewos_dbscan 
tls_fsct <- readLAS("/home/yuchen/segmented_retrain_tls.las")
tls_lewos_graph <- LW_segmentation_graph(tls)
# assign a class base on p_wood
tls_lewos_graph@data[,wood_p := as.numeric(p_wood >= 0.9)]
tls_lewos_graph@data[,wood_s := as.numeric(SoD >= 0.99)]
lidR::plot(tls_lewos_graph,color="wood_p",size=2,colorPalette = c("chartreuse4","cornsilk2"), legend=TRUE)

tls_lewos_dbscan <- LW_segmentation_dbscan(tls)
tls_lewos_dbscan@data[,wood_p := as.numeric(p_wood >= 0.9)]
tls_lewos_dbscan@data[,wood_s := as.numeric(SoD >= 0.99)]
lidR::plot(tls_lewos_dbscan,color="wood_p",size=2,colorPalette = c("chartreuse4","cornsilk2"), legend=TRUE)

# voxR package

# 2. leave=0, wood=1

# True label
tls@data$WL <- replace(tls@data$WL, tls@data$WL==2, 0)
table(tls@data$WL)
lidR::plot(tls,color="WL",size=2,colorPalette = c("chartreuse4","cornsilk2"), legend=TRUE)

# FSCT
tls_fsct@data$label <- replace(tls_fsct@data$label, tls_fsct@data$label==1, 0)
tls_fsct@data$label <- replace(tls_fsct@data$label, tls_fsct@data$label==3, 1)
lidR::plot(tls_fsct,color="label",size=2,colorPalette = c("chartreuse4","cornsilk2"), legend=TRUE)
table(tls_fsct@data$label)

# lewos_graph

table(tls_lewos_graph@data$wood_p)
lidR::plot(tls_lewos_graph,color="wood_p",size=2,colorPalette = c("chartreuse4","cornsilk2"), legend=TRUE)

table(tls_lewos_graph@data$wood_s)
lidR::plot(tls_lewos_graph,color="wood_s",size=2,colorPalette = c("chartreuse4","cornsilk2"), legend=TRUE)

# lewos_dbscan
table(tls_lewos_dbscan@data$wood_p)
lidR::plot(tls_lewos_dbscan,color="wood_p",size=2,colorPalette = c("chartreuse4","cornsilk2"), legend=TRUE)

table(tls_lewos_dbscan@data$wood_s)
lidR::plot(tls_lewos_dbscan,color="wood_s",size=2,colorPalette = c("chartreuse4","cornsilk2"), legend=TRUE)


# 3. start layer analyse
layer_analyse <- function(layer_height){
  l_h <- layer_height
  v_lewos_dbscan_acc <- c()
  v_lewos_dbscan_spe <- c()
  v_lewos_dbscan_rec <- c()
  
  v_fsct_acc <- c()
  v_fsct_spe <- c()
  v_fsct_rec <- c()
  
  method_lewos <- c()
  method_fsct <- c()
  h_min <- min(tls@data$Z)
  h_max <- max(tls@data$Z)
  h_min
  h_max
  
  heights <- c()
  for (i in seq(h_min, h_max-l_h, by=l_h)){
    cat("layer", i, '-', i+l_h, '\t\n')
    
    # True label
    data_tmp <- tls@data[tls@data$Z>=i & tls@data$Z<(i+l_h)]
    
    # lewos_dbscan
    data_tmp2 <- tls_lewos_dbscan@data[tls_lewos_dbscan@data$Z>=i & tls_lewos_dbscan@data$Z<(i+l_h)]
    cat("-> table true tls",table(data_tmp$WL), '\n')
    cat("-> table lewos_dbscan",table(data_tmp2$wood_s), '\n')
    truth <- factor(data_tmp$WL)
    pred <- factor(data_tmp2$wood_s)
    cm <- confusionMatrix(truth, pred)
    list_res <- list("accuracy"=round(as.numeric(cm$overall[1]), 3), "specificity"=round(as.numeric(cm$byClass[2]), 3), "recall"=round(as.numeric(cm$byClass[6]), 3))
    v_lewos_dbscan_acc <- c(v_lewos_dbscan_acc, list_res$accuracy)
    v_lewos_dbscan_spe <- c(v_lewos_dbscan_spe, list_res$specificity)
    v_lewos_dbscan_rec <- c(v_lewos_dbscan_rec, list_res$recall)
    cat("lewos - ok", "\t\n")
    
    # FSCT
    data_tmp2 <- tls_fsct@data[tls_fsct@data$Z>=i & tls_fsct@data$Z<(i+l_h)]
    cat("-> table fsct", table(data_tmp2$label), '\n')
    pred <- factor(data_tmp2$label)
    cat("length(levels(truth)) = ", length(levels(truth)), "\n")
    cat("length(levels(pred)) = ", length(levels(pred)), "\n")
    if (length(levels(truth))==1 & length(levels(pred))==1){
      if (levels(truth) == levels(pred)){
        v_fsct_acc <- c(v_fsct_acc, 1)
        if (levels(pred) == "0"){
          v_fsct_spe <- c(v_fsct_spe, NaN)
          v_fsct_rec <- c(v_fsct_rec, 1)
        }
        if (levels(pred) == "1"){
          v_fsct_spe <- c(v_fsct_spe, 1)
          v_fsct_rec <- c(v_fsct_rec, NaN)
        }
      }
      if (levels(truth) != levels(pred)){
        v_fsct_acc <- c(v_fsct_acc, 0)
        v_fsct_spe <- c(v_fsct_spe, 0)
        v_fsct_rec <- c(v_fsct_rec, 0)
      }
      
    }else if (length(levels(truth))==2 & length(levels(pred)) == 1){
      # length(levels(pred))
      if (levels(pred) == "0"){
        v_fsct_acc <- c(v_fsct_acc, length(truth[truth==0]) / length(pred))
        v_fsct_spe <- c(v_fsct_spe, 0)
        v_fsct_rec <- c(v_fsct_rec, length(truth[truth==0]) / length(pred))
      }
      if (levels(pred) == "1"){
        v_fsct_acc <- c(v_fsct_acc, length(truth[truth==1]) / length(pred))
        v_fsct_spe <- c(v_fsct_spe, length(truth[truth==1]) / length(pred))
        v_fsct_rec <- c(v_fsct_rec, 0)
      }
      
    }else{
      cm <- confusionMatrix(truth, pred)
      list_res <- list("accuracy"=round(as.numeric(cm$overall[1]), 3), "specificity"=round(as.numeric(cm$byClass[2]), 3), "recall"=round(as.numeric(cm$byClass[6]), 3))
      v_fsct_acc <- c(v_fsct_acc, list_res$accuracy)
      v_fsct_spe <- c(v_fsct_spe, list_res$specificity)
      v_fsct_rec <- c(v_fsct_rec, list_res$recall)
    }
    cat("FSCT - ok", "\t\n")
    
    heights <- c(heights, i - h_min + l_h*0.5)
    method_lewos <- c(method_lewos, "lewos_dbscan")
    method_fsct <- c(method_fsct, "fsct")
  }
  
  #heights <- heights + 1.5
  
  
  heights_all <- c(heights,heights)
  method_all <- c(method_lewos,method_fsct)
  acc_all <- c(v_lewos_dbscan_acc, v_fsct_acc)
  spe_all <- c(v_lewos_dbscan_spe, v_fsct_spe)
  rec_all <- c(v_lewos_dbscan_rec, v_fsct_rec)
  
  
  df <- data.frame(Height=heights_all, Method=method_all, Accuracy=acc_all, Specificity=spe_all, Recall=rec_all)
  df
  
  # confidential interval of accuracy
  v_lewos_dbscan_acc <- v_lewos_dbscan_acc[!is.nan(v_lewos_dbscan_acc)]
  v_lewos_dbscan_spe <- v_lewos_dbscan_spe[!is.nan(v_lewos_dbscan_spe)]
  v_lewos_dbscan_rec <- v_lewos_dbscan_rec[!is.nan(v_lewos_dbscan_rec)]
  v_fsct_acc <- v_fsct_acc[!is.nan(v_fsct_acc)]
  v_fsct_spe <- v_fsct_spe[!is.nan(v_fsct_spe)]
  v_fsct_rec <- v_fsct_rec[!is.nan(v_fsct_rec)]
  
  ci_lewos_acc <- (sd(v_lewos_dbscan_acc) * 1.96)/(length(v_lewos_dbscan_acc)**0.5)
  ci_lewos_spe <- (sd(v_lewos_dbscan_spe) * 1.96)/(length(v_lewos_dbscan_spe)**0.5)
  ci_lewos_rec <- (sd(v_lewos_dbscan_rec) * 1.96)/(length(v_lewos_dbscan_rec)**0.5)
  cat("ci_lewos_acc=",ci_lewos_acc,'\n')
  cat("ci_lewos_spe=",ci_lewos_spe,'\n')
  cat("ci_lewos_rec=",ci_lewos_rec,'\n')
  
  ci_fsct_acc <- (sd(v_fsct_acc) * 1.96)/(length(v_fsct_acc)**0.5)
  ci_fsct_spe <- (sd(v_fsct_spe) * 1.96)/(length(v_fsct_spe)**0.5)
  ci_fsct_rec <- (sd(v_fsct_rec) * 1.96)/(length(v_fsct_rec)**0.5)
  cat("ci_fsct_acc=",ci_fsct_acc,'\n')
  cat("ci_fsct_spe=",ci_fsct_spe,'\n')
  cat("ci_fsct_rec=",ci_fsct_rec,'\n')
  
  #####
  
  # Accuracy
  p_acc<- ggplot(df, aes(x = Height, y = Accuracy, colour = Method)) +
    geom_line(lwd=.7) +
    geom_ribbon(data=df[df$Method=="lewos_dbscan",], aes(ymin = Accuracy - ci_lewos_acc, ymax = Accuracy+ ci_lewos_acc), 
                alpha = .3, fill = "darkseagreen3", color = "transparent") +
    geom_ribbon(data=df[df$Method=="fsct",], aes(ymin = Accuracy - ci_fsct_acc, ymax = Accuracy+ ci_fsct_acc), 
                alpha = .3, fill = "darkorange1", color = "transparent") +
    labs(x = "Heights", y = "Accuracy (95% CI)") +
    ylim(-0.15,1.15)+
    scale_color_manual(name = "Method", values =c('lewos_dbscan'='aquamarine4','fsct'='chocolate3'), labels = c('fsct','lewos')) + 
    coord_flip() +
    theme(text = element_text(size=15), axis.text=element_text(size=12), legend.title = element_text(size=15), legend.text = element_text(size=14))
  #p_acc
  
  # Specificity
  p_spe<- ggplot(df, aes(x = Height, y = Specificity, colour = Method)) +
    geom_line(lwd=.7) +
    geom_ribbon(data=df[df$Method=="lewos_dbscan",], aes(ymin = Specificity - ci_lewos_spe, ymax = Specificity+ ci_lewos_spe), 
                alpha = .3, fill = "darkseagreen3", color = "transparent") +
    geom_ribbon(data=df[df$Method=="fsct",], aes(ymin = Specificity - ci_fsct_spe, ymax = Specificity+ ci_fsct_spe), 
                alpha = .3, fill = "darkorange1", color = "transparent") +
    labs(x = "Heights", y = "Specificity (95% CI)") +
    ylim(-0.15,1.15)+
    scale_color_manual(name = "Method", values =c('lewos_dbscan'='aquamarine4','fsct'='chocolate3'), labels = c('fsct','lewos')) + 
    coord_flip() +
    theme(text = element_text(size=15), axis.text=element_text(size=12), legend.title = element_text(size=15), legend.text = element_text(size=14))
  #p_spe
  
  # Recall
  p_rec<- ggplot(df, aes(x = Height, y = Recall, colour = Method)) +
    geom_line(lwd=.7) +
    geom_ribbon(data=df[df$Method=="lewos_dbscan",], aes(ymin = Recall - ci_lewos_rec, ymax = Recall+ ci_lewos_rec), 
                alpha = .3, fill = "darkseagreen3", color = "transparent") +
    geom_ribbon(data=df[df$Method=="fsct",], aes(ymin = Recall - ci_fsct_rec, ymax = Recall+ ci_fsct_rec), 
                alpha = .3, fill = "darkorange1", color = "transparent") +
    labs(x = "Heights", y = "Recall (95% CI)") +
    ylim(-0.15,1.15)+
    scale_color_manual(name = "Method", values =c('lewos_dbscan'='aquamarine4','fsct'='chocolate3'), labels = c('fsct','lewos')) + 
    coord_flip() +
    theme(text = element_text(size=15), axis.text=element_text(size=12), legend.title = element_text(size=15), legend.text = element_text(size=14))
  #p_rec
  my_list <- list("acc" = p_acc, "spe" = p_spe, "rec" = p_rec)
  return(my_list)
}

res <- layer_analyse(layer_height = 1)
res$acc
res$spe
res$rec