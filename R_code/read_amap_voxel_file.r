library("AMAPVox")
vx <- readVoxelSpace("/home/yuchen/Documents/PhD/data_for_project/22-05-16_vol_trajet_new_vol_data_after_icp/test_sans_weights.vox")









## plot ###
p1<- ggplot(df, aes(x = Height, y = Accuracy)) +
  geom_ribbon(aes(ymin = Accuracy - ci_lewos_accuracy, ymax = Accuracy+ ci_lewos_accuracy), 
              alpha = .3, fill = "darkseagreen3", color = "transparent") +
  geom_line(color = "aquamarine4",  lwd=.7) +
  labs(x = "Heights", y = "Accuracy") +
  scale_color_manual(name = "Method", values = method) + 
  coord_flip()
p1