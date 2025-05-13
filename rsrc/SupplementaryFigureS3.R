install.packages("ggplot2")
install.packages("readxl")

library(ggplot2)
library(readxl)

setwd('/Users/chooyujin/OneDrive/work/2024/Phospho/paper_figure_2025/')

methylR_df <- read_xlsx('SAPPmethylR_test_metricResult.xlsx')
phosphoY_df <- read_xlsx('SAPPphosY_test_metricResult.xlsx')
acetylK_df <- read_xlsx('SAPPacetylK_test_metricResult.xlsx')
methylK_df <- read_xlsx('SAPPmethylK_test_metricResult.xlsx')
sumoK_df <- read_xlsx('SAPPsumoK_test_metricResult.xlsx')
ubiquitinK_df <- read_xlsx('SAPPubiquitinK_test_metricResult.xlsx')

methylR_df$Specificity <- as.factor(methylR_df$Specificity)
phosphoY_df$Specificity <- as.factor(phosphoY_df$Specificity)
acetylK_df$Specificity <- as.factor(acetylK_df$Specificity)
methylK_df$Specificity <- as.factor(methylK_df$Specificity)
sumoK_df$Specificity <- as.factor(sumoK_df$Specificity)
ubiquitinK_df$Specificity <- as.factor(ubiquitinK_df$Specificity)

methylR_p<-ggplot(methylR_df, aes(x = Metric, y = Score, group = interaction(Model, Specificity),
                            color = Model, linetype = Specificity)) +
  geom_line(size = 1) +  
  geom_point(size = 2) + 
  labs(title = "R Methylation [5230/1301]",
       x = "Metric", y = "Score",
       color = "Model", linetype = "Specificity") +
  theme_minimal()+  
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  +  
  scale_color_manual(values = c("Fine-tuning" = "#E31A1C",
                                "MusiteDeep" = "#608BC1", 
                                "Transfer" ="#8B8989"
  ))

phosphoY_p<-ggplot(phosphoY_df, aes(x = Metric, y = Score, group = interaction(Model, Specificity),
                                  color = Model, linetype = Specificity)) +
  geom_line(size = 1) +  
  geom_point(size = 2) + 
  labs(title = "Y Phosphorylation [10113/2529]",
       x = "Metric", y = "Score",
       color = "Model", linetype = "Specificity") +
  theme_minimal()+  
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  +  
  scale_color_manual(values = c("Fine-tuning" = "#E31A1C",
                                "MusiteDeep" = "#608BC1", 
                                "Transfer" ="#8B8989"
  ))

acetylK_p<-ggplot(acetylK_df, aes(x = Metric, y = Score, group = interaction(Model, Specificity),
                                  color = Model, linetype = Specificity)) +
  geom_line(size = 1) +  
  geom_point(size = 2) + 
  labs(title = "K Acetylation [19214/4804]",
       x = "Metric", y = "Score",
       color = "Model", linetype = "Specificity") +
  theme_minimal()+  
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  +  
  scale_color_manual(values = c("Fine-tuning" = "#E31A1C",
                                "MusiteDeep" = "#608BC1", 
                                "Transfer" ="#8B8989"
  ))

methylK_p<-ggplot(methylK_df, aes(x = Metric, y = Score, group = interaction(Model, Specificity),
                                  color = Model, linetype = Specificity)) +
  geom_line(size = 1) +  
  geom_point(size = 2) + 
  labs(title = "K Methylation [1504/376]",
       x = "Metric", y = "Score",
       color = "Model", linetype = "Specificity") +
  theme_minimal()+  
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  +  
  scale_color_manual(values = c("Fine-tuning" = "#E31A1C",
                                "MusiteDeep" = "#608BC1", 
                                "Transfer" ="#8B8989"
  ))

sumoK_p<-ggplot(sumoK_df, aes(x = Metric, y = Score, group = interaction(Model, Specificity),
                                  color = Model, linetype = Specificity)) +
  geom_line(size = 1) +  
  geom_point(size = 2) + 
  labs(title = "K SUMOylation [16574/4144]",
       x = "Metric", y = "Score",
       color = "Model", linetype = "Specificity") +
  theme_minimal()+  
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  +  
  scale_color_manual(values = c("Fine-tuning" = "#E31A1C",
                                "MusiteDeep" = "#608BC1", 
                                "Transfer" ="#8B8989"
  ))

ubiquitinK_p<-ggplot(ubiquitinK_df, aes(x = Metric, y = Score, group = interaction(Model, Specificity),
                                  color = Model, linetype = Specificity)) +
  geom_line(size = 1) +  
  geom_point(size = 2) + 
  labs(title = "K Ubiquitination [3796/950]",
       x = "Metric", y = "Score",
       color = "Model", linetype = "Specificity") +
  theme_minimal()+  
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  +  
  scale_color_manual(values = c("Fine-tuning" = "#E31A1C",
                                "MusiteDeep" = "#608BC1", 
                                "Transfer" ="#8B8989"
  ))

