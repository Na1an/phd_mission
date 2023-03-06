library("ggplot2")
library("lattice")
library("caret")

truth <- c(0,1,1,0)
pred <- c(1,1,1,0)
truth <- factor(truth)
pred <- factor(pred)

# pred 在前，true label在后
# and 1 is specificity
cm <- confusionMatrix(pred, truth)
cm