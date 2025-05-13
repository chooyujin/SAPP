install.packages("ggplot2")
install.packages("ROCR")

library(ggplot2)
library(ROCR)

setwd('/Users/chooyujin/OneDrive/work/2024/Phospho/paper_figure_2025/')

pred_df = read.csv('Feature_selection_10fold_result.csv')
RSA <- pred_df[pred_df$Model == 'with RSA',]
onlySeq <- pred_df[pred_df$Model == 'only sequence',]
pLDDT <- pred_df[pred_df$Model == 'with pLDDT',]
Disorder <- pred_df[pred_df$Model == 'with disorder',]

RSA_pred <- prediction(RSA$Pred, RSA$Label)
onlySeq_pred <- prediction(onlySeq$Pred, onlySeq$Label)
pLDDT_pred <- prediction(pLDDT$Pred, pLDDT$Label)
Disorder_pred <- prediction(Disorder$Pred, Disorder$Label)

RSA_perf <- performance(RSA_pred, "tpr", "fpr")
onlySeq_perf <- performance(onlySeq_pred, "tpr", "fpr")
pLDDT_perf <- performance(pLDDT_pred, "tpr", "fpr")
Disorder_perf <- performance(Disorder_pred, "tpr", "fpr")

truncate4 <- function(x) floor(x * 1e4) / 1e4

RSA_auc <- performance(RSA_pred, "auc")@y.values[[1]]
RSA_auc <- truncate4(RSA_auc)
onlySeq_auc <- performance(onlySeq_pred, "auc")@y.values[[1]]
onlySeq_auc <- truncate4(onlySeq_auc)
Disorder_auc <- performance(Disorder_pred, "auc")@y.values[[1]]
Disorder_auc <- truncate4(Disorder_auc)
pLDDT_auc <- performance(pLDDT_pred, "auc")@y.values[[1]]
pLDDT_auc <- truncate4(pLDDT_auc)

RSA_roc <- data.frame(
  sensitivity = RSA_perf@y.values[[1]],
  specificity = RSA_perf@x.values[[1]],
  model = sprintf("with RSA (AUC=%.4f)", RSA_auc)
)
onlySeq_roc <- data.frame(
  sensitivity = onlySeq_perf@y.values[[1]],
  specificity = onlySeq_perf@x.values[[1]],
  model = sprintf("only sequence (AUC=%.4f)", onlySeq_auc)
)
Disorder_roc <- data.frame(
  sensitivity = Disorder_perf@y.values[[1]],
  specificity = Disorder_perf@x.values[[1]],
  model = sprintf("with disorder (AUC=%.4f)", Disorder_auc)
)
pLDDT_roc <- data.frame(
  sensitivity = pLDDT_perf@y.values[[1]],
  specificity = pLDDT_perf@x.values[[1]],
  model = sprintf("with pLDDT (AUC=%.4f)", pLDDT_auc)
)


color_labels <- c(
  unique(RSA_roc$model),
  unique(onlySeq_roc$model),
  unique(pLDDT_roc$model),
  unique(Disorder_roc$model)
)

color_values <-  c("#E31A1C", "#FD8D3C", "#FED976", "#40AD5A")
names(color_values) <- color_labels

# ggplot
p1 <- ggplot() +
  geom_line(data = RSA_roc, aes(x = specificity, y = sensitivity, color = model), linetype = "solid", size = 0.7) +
  geom_line(data = onlySeq_roc, aes(x = specificity, y = sensitivity, color = model), linetype = "solid", size = 0.7) +
  geom_line(data = Disorder_roc, aes(x = specificity, y = sensitivity, color = model), linetype = "solid", size = 0.7) +
  geom_line(data = pLDDT_roc, aes(x = specificity, y = sensitivity, color = model), linetype = "solid", size = 0.7) +
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
  geom_line(data = RSA_roc, aes(x = specificity, y = sensitivity, color = model), linetype = "solid", size = 0.7) +
  geom_line(data = onlySeq_roc, aes(x = specificity, y = sensitivity, color = model), linetype = "solid", size = 0.7) +
  geom_line(data = Disorder_roc, aes(x = specificity, y = sensitivity, color = model), linetype = "solid", size = 0.7) +
  geom_line(data = pLDDT_roc, aes(x = specificity, y = sensitivity, color = model), linetype = "solid", size = 0.7) +
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

