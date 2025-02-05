library(survminer)
library(survival)

df <- read.csv("your-time-event-file.csv")
colnames(df)
df$ClassTocompare <- factor(df$ClassToCompare, levels = c("group1", "group2", "group3", "group4"))
surv.object<-Surv(df$time, df$event)

km=survfit(formula=surv.object ~ ClassTocompare,data = df)

tiff("km-curve.tiff", width=17.75, height=17.75, units="cm", res=300)
ggsurvplot(km,break.time.by = 12,xlim = c(0,60),risk.table = TRUE, pval=TRUE, pval.method = TRUE, palette= c('#4F95DB', '#DB2F36','#D6B709', '#187D2A'),legend=c(0.8,0.2),legend.labs=c("group1", "group2", "group3", "group4"))
dev.off()