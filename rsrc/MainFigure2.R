install.packages("ggplot2")
install.packages("ROCR")

library(ggplot2)
library(ROCR)

setwd('/Users/chooyujin/OneDrive/work/2024/Phospho/paper_figure_2025/')

pred_df = read.csv('total_model_test_result.csv')
SAPP <- pred_df[pred_df$Model == 'SAPPphos',]
MusiteDeep <- pred_df[pred_df$Model == 'MusiteDeep',]
DeepPSP <- pred_df[pred_df$Model == 'DeepPSP',]
DeepPhos <- pred_df[pred_df$Model == 'DeepPhos',]

SAPP_pred <- prediction(SAPP$Pred, SAPP$Label)
Musitedeep_pred <- prediction(MusiteDeep$Pred, MusiteDeep$Label)
DeepPSP_pred <- prediction(DeepPSP$Pred, DeepPSP$Label)
DeepPhos_pred <- prediction(DeepPhos$Pred, DeepPhos$Label)

SAPP_perf <- performance(SAPP_pred, "tpr", "fpr")
Musitedeep_perf <- performance(Musitedeep_pred, "tpr", "fpr")
DeepPSP_perf <- performance(DeepPSP_pred, "tpr", "fpr")
DeepPhos_perf <- performance(DeepPhos_pred, "tpr", "fpr")

truncate4 <- function(x) floor(x * 1e4) / 1e4

SAPP_auc <- performance(SAPP_pred, "auc")@y.values[[1]]
SAPP_auc <- truncate4(SAPP_auc)
Musitedeep_auc <- performance(Musitedeep_pred, "auc")@y.values[[1]]
Musitedeep_auc <- truncate4(Musitedeep_auc)
DeepPSP_auc <- performance(DeepPSP_pred, "auc")@y.values[[1]]
DeepPSP_auc <- truncate4(DeepPSP_auc)
DeepPhos_auc <- performance(DeepPhos_pred, "auc")@y.values[[1]]
DeepPhos_auc <- truncate4(DeepPhos_auc)

SAPP_roc <- data.frame(
  sensitivity = SAPP_perf@y.values[[1]],
  specificity = SAPP_perf@x.values[[1]],
  model = sprintf("SAPPphos (AUC=%.4f)", SAPP_auc)
)
Musitedeep_roc <- data.frame(
  sensitivity = Musitedeep_perf@y.values[[1]],
  specificity = Musitedeep_perf@x.values[[1]],
  model = sprintf("MusiteDeep (AUC=%.4f)", Musitedeep_auc)
)
DeepPSP_roc <- data.frame(
  sensitivity = DeepPSP_perf@y.values[[1]],
  specificity = DeepPSP_perf@x.values[[1]],
  model = sprintf("DeepPSP (AUC=%.4f)", DeepPSP_auc)
)
DeepPhos_roc <- data.frame(
  sensitivity = DeepPhos_perf@y.values[[1]],
  specificity = DeepPhos_perf@x.values[[1]],
  model = sprintf("DeepPhos (AUC=%.4f)", DeepPhos_auc)
)

color_labels <- c(
  unique(SAPP_roc$model),
  unique(Musitedeep_roc$model),
  unique(DeepPhos_roc$model),
  unique(DeepPSP_roc$model)
)

color_values <-  c("#E31A1C", "#FD8D3C", "#FED976", "#40AD5A")
names(color_values) <- color_labels

p1 <- ggplot() +
  geom_line(data = SAPP_roc, aes(x = specificity, y = sensitivity, color = model), linetype = "solid", size = 0.7) +
  geom_line(data = Musitedeep_roc, aes(x = specificity, y = sensitivity, color = model), linetype = "solid", size = 0.7) +
  geom_line(data = DeepPhos_roc, aes(x = specificity, y = sensitivity, color = model), linetype = "solid", size = 0.7) +
  geom_line(data = DeepPSP_roc, aes(x = specificity, y = sensitivity, color = model), linetype = "solid", size = 0.7) +
  labs(x = "1 - Specificity", y = "Sensitivity", color = "Model") +
  theme(panel.background = element_rect(fill = "white"),
        axis.line.x = element_line(color = 'black'),
        axis.line.y = element_line(color = 'black'),
        axis.title = element_text(size = 11, face = "bold"),
        panel.grid.major = element_line(color = "gray", linetype = "dotted"),
        panel.grid.minor = element_line(color = "gray", linetype = "dotted")) +
  scale_x_continuous(expand = c(0.0, 0.0), limits = c(0, 1.00)) +
  scale_y_continuous(expand = c(0.0, 0.0), limits = c(0, 1.01)) +
  scale_color_manual(values = color_values) +
  theme(legend.position = c(0.93, 0.06),
        legend.justification = c(0.90, 0.10),
        legend.margin = margin(0, 0, 0, 0),
        legend.box.background = element_rect(color = 'black'),
        legend.box.margin = margin(2, 2, 2, 2),
        panel.border = element_rect(color = "black", fill = NA, size = 0.8),
        legend.text = element_text(size = 7),
        legend.title = element_text(size = 8))

p2 <- ggplot() +
  geom_line(data = SAPP_roc, aes(x = specificity, y = sensitivity, color = model), linetype = "solid", size = 0.7) +
  geom_line(data = Musitedeep_roc, aes(x = specificity, y = sensitivity, color = model), linetype = "solid", size = 0.7) +
  geom_line(data = DeepPhos_roc, aes(x = specificity, y = sensitivity, color = model), linetype = "solid", size = 0.7) +
  geom_line(data = DeepPSP_roc, aes(x = specificity, y = sensitivity, color = model), linetype = "solid", size = 0.7) +
  labs(x = "1 - Specificity", y = "Sensitivity", color = "Model") + 
  theme(panel.background = element_rect(fill = "white"),
        axis.line.x = element_blank(), 
        axis.line.y = element_blank(),
        panel.border = element_rect(color = "black", fill = NA, size = 1.0),
        axis.title = element_text(size = 11, face = "bold"),
        panel.grid.major = element_line(color = "gray", linetype = "dotted"),
        panel.grid.minor = element_line(color = "gray", linetype = "dotted"),
        legend.position = "none") +
  scale_x_continuous(expand = c(0.0, 0.0), limits = c(0, 0.2)) + 
  scale_y_continuous(expand = c(0.0, 0.0), limits = c(0.5, 1.01)) + 
  coord_fixed(ratio = 0.8) +
  scale_color_manual(values = color_values)

