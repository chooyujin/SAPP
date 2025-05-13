getwd()
setwd('/Users/chooyujin/OneDrive/work/2024/Phospho/paper_figure_2025/')

library(ggplot2)
library(RColorBrewer)

data = read.csv('phospho_logodds.csv')


colors = c("#E88471","#EEB479","#E9E29C","#9CCB86","#009392")
p<- ggplot(data, aes(y=Metric, x=Odd_ratio,fill=Metric)) + 
  geom_bar(stat = 'identity', width=0.7) + 
  labs(x='log odds ratio',y=NULL) + 
  theme(panel.background = element_rect(fill = "white"),
        panel.grid.major = element_line(color = "grey", linetype = "dotted"), 
        axis.line.x = element_line(color='black'),
        axis.line.y = element_line(color='black'),
        axis.text.y = element_text(size=11,color='black'),
        axis.text.x = element_text(size=11,color='black'),
        axis.title = element_text(size = 12, face = "bold"),
        panel.grid.minor = element_blank()) + 
  scale_x_continuous(expand=c(0.0,0.0),limits =c(0.0,2.1))+ 
  guides(fill="none")+
  scale_fill_manual(values=colors)

