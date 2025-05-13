install.packages("ggplot2")
install.packages("ggpubr")

library(ggplot2)
library(ggpubr)

setwd('/Users/chooyujin/OneDrive/work/2024/Phospho/paper_figure_2025/')

predicted_str_df = read.csv('predicted_structure_quality_andtotal.csv')

pos_df <- predicted_str_df[predicted_str_df$Label == 1,]

model_order <- c("SAPPphos", "DeepPhos","DeepPSP", "MusiteDeep")
pos_df$Model <- factor(pos_df$Model, levels = model_order)

pos_df$TMscore <- factor(pos_df$TMscore, levels = c("low", "high", "Total"))

comparison_list <- list(c("low", "high"))

p <- ggplot(pos_df, aes(x = Model, y = Prob, fill = TMscore)) +
  geom_boxplot() +
  scale_fill_manual(values = c("#9CCB86", "#009392", "#E69F00")) +  # A, B, C 색상 추가
  xlab("Model") +
  ylab("Prob") + 
  theme_bw() + 
  theme(panel.background = element_rect(fill = "white")) + 
  stat_compare_means(method = "wilcox.test", 
                     comparisons = comparison_list, 
                     aes(group = TMscore),  # TMscore 그룹을 기준으로 비교
                     label = "p.signif",  # p-value를 별표로 표시 (*, **, ***)
                     group.by = "Model",  # 각 Model별로 A, B 비교
                     label.y = 1.03) +  
  theme(legend.box.background=element_rect(color='black'),
        legend.box.margin = margin(2, 2, 2, 2),
        axis.title = element_text(size = 15, face = "bold"),
        panel.border = element_rect(color = "black", fill = NA, size = 1.2))
p



