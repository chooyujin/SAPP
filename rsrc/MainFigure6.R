install.packages("ggplot2")
install.packages("readxl")

library(ggplot2)
library(readxl)

setwd('/Users/chooyujin/OneDrive/work/2024/Phospho/paper_figure_2025/')

CMGC_df <- read_xlsx('CMGC_test_metricResult.xlsx')
CAMK_df <- read_xlsx('CAMK_test_metricResult.xlsx')
CDK_df <- read_xlsx('CDK_test_metricResult.xlsx')
AGC_df <- read_xlsx('AGC_test_metricResult.xlsx')
MAPK_df <- read_xlsx('MAPK_test_metricResult.xlsx')
PKA_df <- read_xlsx('PKA_test_metricResult.xlsx')
PKC_df <- read_xlsx(('PKC_test_metricResult.xlsx'))
CK2_df <- read_xlsx('CK2_test_metricResult.xlsx')

CMGC_df$Specificity <- as.factor(CMGC_df$Specificity)
CAMK_df$Specificity <- as.factor(CAMK_df$Specificity)
CDK_df$Specificity <- as.factor(CDK_df$Specificity)
AGC_df$Specificity <- as.factor(AGC_df$Specificity)
MAPK_df$Specificity <- as.factor(MAPK_df$Specificity)
PKA_df$Specificity <- as.factor(PKA_df$Specificity)
PKC_df$Specificity <- as.factor(PKC_df$Specificity)
CK2_df$Specificity <- as.factor(CK2_df$Specificity)

CMGC_p<-ggplot(CMGC_df, aes(x = Metric, y = Score, group = interaction(Model, Specificity),
                  color = Model, linetype = Specificity)) +
  geom_line(size = 1) +  
  geom_point(size = 2) + 
  labs(title = "CMGC Kinase [8280/2070]",
       x = "Metric", y = "Score",
       color = "Model", linetype = "Specificity") +
  theme_minimal()+  
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  +  
  scale_color_manual(values = c("Transfer" = "#E31A1C",
                                "DeepPhos" = "#608BC1", 
                                "From Scratch" ="#8B8989"
  ),
  limits = c("Transfer","DeepPhos","From Scratch"))

CAMK_p<-ggplot(CAMK_df, aes(x = Metric, y = Score, group = interaction(Model, Specificity),
                       color = Model, linetype = Specificity)) +
  geom_line(size = 1) +  
  geom_point(size = 2) + 
  labs(title = "CAMK Kinase [2756/690]",
       x = "Metric", y = "Score",
       color = "Model", linetype = "Specificity") +
  theme_minimal()+  
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  +  
  scale_color_manual(values = c("Transfer" = "#E31A1C",
                                "DeepPhos" = "#608BC1", 
                                "From Scratch" ="#8B8989"
  ),
  limits = c("Transfer","DeepPhos","From Scratch"))

CDK_p<-ggplot(CDK_df, aes(x = Metric, y = Score, group = interaction(Model, Specificity),
                       color = Model, linetype = Specificity)) +
  geom_line(size = 1) +  
  geom_point(size = 2) + 
  labs(title = "CDK Kinase [1938/488]",
       x = "Metric", y = "Score",
       color = "Model", linetype = "Specificity") +
  theme_minimal()+  
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  +  
  scale_color_manual(values = c("Transfer" = "#E31A1C",
                                "DeepPhos" = "#608BC1", 
                                "From Scratch" ="#8B8989"
  ),
  limits = c("Transfer","DeepPhos","From Scratch"))

AGC_p<-ggplot(AGC_df, aes(x = Metric, y = Score, group = interaction(Model, Specificity),
                       color = Model, linetype = Specificity)) +
  geom_line(size = 1) +  
  geom_point(size = 2) + 
  labs(title = "AGC Kinase [6774/1694]",
       x = "Metric", y = "Score",
       color = "Model", linetype = "Specificity") +
  theme_minimal()+  
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  +  
  scale_color_manual(values = c("Transfer" = "#E31A1C",
                                "DeepPhos" = "#608BC1", 
                                "From Scratch" ="#8B8989"
  ),
  limits = c("Transfer","DeepPhos","From Scratch"))

MAPK_p<-ggplot(MAPK_df, aes(x = Metric, y = Score, group = interaction(Model, Specificity),
                       color = Model, linetype = Specificity)) +
  geom_line(size = 1) +  
  geom_point(size = 2) + 
  labs(title = "MAPK Kinase [1616/404]",
       x = "Metric", y = "Score",
       color = "Model", linetype = "Specificity") +
  theme_minimal()+  
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  +  
  scale_color_manual(values = c("Transfer" = "#E31A1C",
                                "DeepPhos" = "#608BC1", 
                                "From Scratch" ="#8B8989"
  ),
  limits = c("Transfer","DeepPhos","From Scratch"))

PKA_p<-ggplot(PKA_df, aes(x = Metric, y = Score, group = interaction(Model, Specificity),
                       color = Model, linetype = Specificity)) +
  geom_line(size = 1) +  
  geom_point(size = 2) + 
  labs(title = "PKA Kinase [1632/408]",
       x = "Metric", y = "Score",
       color = "Model", linetype = "Specificity") +
  theme_minimal()+  
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  +  
  scale_color_manual(values = c("Transfer" = "#E31A1C",
                                "DeepPhos" = "#608BC1", 
                                "From Scratch" ="#8B8989"
  ),
  limits = c("Transfer","DeepPhos","From Scratch"))

PKC_p<-ggplot(PKC_df, aes(x = Metric, y = Score, group = interaction(Model, Specificity),
                       color = Model, linetype = Specificity)) +
  geom_line(size = 1) +  
  geom_point(size = 2) + 
  labs(title = "PKC Kinase [1380/346]",
       x = "Metric", y = "Score",
       color = "Model", linetype = "Specificity") +
  theme_minimal()+  
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  +  
  scale_color_manual(values = c("Transfer" = "#E31A1C",
                                "DeepPhos" = "#608BC1", 
                                "From Scratch" ="#8B8989"
  ),
  limits = c("Transfer","DeepPhos","From Scratch"))

CK2_p<-ggplot(CMGC_df, aes(x = Metric, y = Score, group = interaction(Model, Specificity),
                       color = Model, linetype = Specificity)) +
  geom_line(size = 1) +  
  geom_point(size = 2) + 
  labs(title = "CK2 Kinase [1137/285]",
       x = "Metric", y = "Score",
       color = "Model", linetype = "Specificity") +
  theme_minimal()+  
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  +  
  scale_color_manual(values = c("Transfer" = "#E31A1C",
                                "DeepPhos" = "#608BC1", 
                                "From Scratch" ="#8B8989"
  ),
  limits = c("Transfer","DeepPhos","From Scratch"))

