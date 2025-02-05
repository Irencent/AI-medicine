##################
#forest multivariate shh revision
library(forestploter)
df <- read.csv("./forest_plot_cox_multivariate_shh_revision.csv")
df$Subgroup <- ifelse(is.na(df$est), 
                      df$Subgroup,
                      paste0("   ", df$Subgroup))
df$se <- (log(df$high) - log(df$est))/1.96
df$` ` <- paste(rep(" ", 20), collapse = " ")


df$`HR (95% CI)` <- ifelse(is.na(df$est), "",ifelse(df$est==1, "Reference",
                                                    sprintf("%.2f (%.2f to %.2f)",
                                                            df$est, df$low, df$high)))

df$N <- ifelse(is.na(df$est),"",df$N)
df$p.value <- ifelse((is.na(df$est))|(df$est==1),"",df$p.value)
df
tiff("forest_multivariate_shh_revision.tiff", width=19, height=11, units="cm", res=300)

p <- forest(df[,c(1:2,8:9,6)],
            est = df$est,
            lower = df$low, 
            upper = df$high,
            sei = df$se,
            ci_column = 3,
            ref_line = 1,
            x_trans = c("log"),
            xlog = TRUE,   
            arrow_lab = c("", ""),
            xlim = c(0,5.5),
            ticks_at = c(0.5, 1, 2,3,4,5),
            footnote = "")


# Print plot
plot(p)

dev.off()

#molecular subtypes
library(survminer)
library(survival)

df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
surv.object<-Surv(df$OS_nomissing_adjusted, df$mortality)

#g3&g4 combined
km=survfit(formula=surv.object ~ mol_sub_1,data = df)

tiff("km_g_combined_revision.tiff", width=17.75, height=17.75, units="cm", res=300)
ggsurvplot(km,break.time.by = 12,xlim = c(0,60),risk.table = TRUE,pval=TRUE, pval.method = TRUE, palette= c('#5ABBC0','#C5C6C7','#B92B78','#6197D5'),legend=c(0.8,0.2),legend.labs=c("non-WNT/non-SHH","SHH TP53-","SHH TP53+","WNT"))
dev.off()

#shh combined
df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )

df <- df %>% 
  mutate(mol_sub_3 = factor(mol_sub_2, levels = c("SHH","G3","G4","WNT")))
km=survfit(formula=surv.object ~ mol_sub_3,data = df)

tiff("km_shh_combined_revision.tiff", width=17.75, height=17.75, units="cm", res=300)
ggsurvplot(km,break.time.by = 12,xlim = c(0,60),risk.table = TRUE,pval=TRUE, pval.method = TRUE, palette= c('#F1564D','#F19E38','#80BA5B','#6197D5'),legend=c(0.8,0.2),legend.labs=c("SHH","G3","G4","WNT"))
dev.off()

#g3&g4
df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
df<-subset(df, df[,"mol_sub"]== "G3"|df[,"mol_sub"]== "G4")
surv.object<-Surv(df$OS_nomissing_adjusted, df$mortality)
km=survfit(formula=surv.object ~ mol_sub,data = df)

tiff("km_g3_g4_revision.tiff", width=17.75, height=17.75, units="cm", res=300)
ggsurvplot(km,break.time.by = 12,xlim = c(0,60),risk.table = TRUE,pval=TRUE, pval.method = TRUE, palette= c('#F19E38','#80BA5B'),legend=c(0.8,0.2),legend.labs=c("G3","G4"))
dev.off()

#shh
df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
df<-subset(df, df[,"mol_sub"]== "SHHTP53-"|df[,"mol_sub"]== "SHHTP53+")
surv.object<-Surv(df$OS_nomissing_adjusted, df$mortality)
km=survfit(formula=surv.object ~ mol_sub,data = df)

tiff("km_shh_revision.tiff", width=17.75, height=17.75, units="cm", res=300)
ggsurvplot(km,break.time.by = 12,xlim = c(0,120),risk.table = TRUE,pval=TRUE, pval.method = TRUE, palette= c('#C5C6C7','#B92B78'),legend=c(0.8,0.2),legend.labs=c("SHH TP53-","SHH TP53+"))
dev.off()

library(npsurv)
library(rms)
km.by<-npsurv(surv.object ~ mol_sub_2,data = df,
              conf.type = "log-log"
              )
survplot(km.by,conf.int = 0.95,
         lty = c(2,1), col = c("black","gray47"), xlab = "Time", ylab = "Survival probability",
         xlim = c(0,120),time.inc = 24, label.curves = T, n.risk = F)

