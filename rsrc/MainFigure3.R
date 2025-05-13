install.packages("ggplot2")
install.packages("ROCR")
install.packages("ggpubr")

library(ggplot2)
library(ROCR)
library(ggpubr)

setwd('/Users/chooyujin/OneDrive/work/2024/Phospho/paper_figure_2025/')

pred_df = read.csv('SAPPphos_with_without_RSA_result.csv')
pred_df$Label[pred_df$Label == 1] <- "Phosphorylated"
pred_df$Label[pred_df$Label == 0] <- "Non-Phosphorylated"

with_RSA <- pred_df[pred_df$Feature == 'with RSA feature',]
without_RSA <- pred_df[pred_df$Feature == 'without RSA feature',]

pos_df <- pred_df[pred_df$Label == "Phosphorylated",]
neg_df <- pred_df[pred_df$Label == "Non-Phosphorylated",]

model_order <- c("with RSA feature", "without RSA feature")
pos_df$Feature <- factor(pos_df$Feature, levels = model_order)

pred_df$Label <- as.character(pred_df$Label)

p<-ggplot(pred_df, aes(x = Label, y = Prob,fill=Feature)) +
  geom_boxplot() +
  scale_fill_manual(values = c("#608BC1","#CBDCEB")) + 
  xlab("Label") +
  ylab("Prob") + 
  theme_bw() + 
  theme(panel.background = element_rect(fill = "white"))+ 
  stat_compare_means(method = "wilcox.test", label.y = c(1.03,1.03)) + 
  theme(legend.box.background=element_rect(color='black'),
        legend.box.margin = margin(2, 2, 2, 2),
        axis.title = element_text(size = 15, face = "bold"),
        panel.border = element_rect(color = "black", fill = NA, size = 1.2))

p
