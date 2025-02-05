calc_p <- function(c1, c2=c(1, 6, 2, 21), c3, require_c3=F) {
  if (require_c3) {
    c4 <- c2 - c3 - c1
  }else{
    c4 <- c2 - c1
  }
  tab <- rbind(c1, c4)
  print(tab)
  print(fisher.test(tab))
}
calc_p(c1=c(4, 40, 20, 62), c2=c(115, 204, 95, 319), require_c3=F)
calc_p(c1=c(0, 3, 0, 15), c2=c(1, 6, 2, 21))
calc_p(c1=c(1, 3, 2, 5), c2=c(1, 6, 2, 21), c3= c(0 , 0 ,0, 1), require_c3=T)
calc_p(c1=c(0, 0, 0, 1), c2=c(1, 6, 2, 21))
calc_p(c1=c(1, 6, 2, 21), c2=c(4, 40, 20, 62))

calc_p(c1=c(0, 1, 0, 7), c2=c(2, 9, 8, 22))
calc_p(c1=c(1, 4, 6, 12), c2=c(2, 9, 8, 22), c3= c(1, 1, 0, 2), require_c3=T)
calc_p(c1=c(0, 3, 2, 1), c2=c(2, 9, 8, 22), c3= c(1, 1, 0, 2), require_c3=T)
calc_p(c1=c(1, 1, 0, 2), c2=c(2, 9, 8, 22))
calc_p(c1=c(2, 9, 8, 22), c2=c(4, 40, 20, 62))
calc_p(c1=c(2, 13, 8, 31), c2=c(4, 40, 20, 62))
calc_p(c1=c(0, 3, 0, 16), c2=c(4, 40, 20, 62))

calc_p(c1=c(0, 1, 0, 0), c2=c(1, 13, 3, 12))
calc_p(c1=c(1, 5, 1, 5), c2=c(1, 13, 3, 12), c3= c(0 , 2 ,1, 4), require_c3=T)
calc_p(c1=c(0, 5, 1, 3), c2=c(1, 13, 3, 12), c3= c(0 , 2 ,1, 4), require_c3=T)
calc_p(c1=c(0 , 2 ,1, 4), c2=c(1, 13, 3, 12))
calc_p(c1=c(1, 13, 3, 12), c2=c(4, 40, 20, 62))

calc_p(c1=c(0, 1, 0, 0), c2=c(1, 21, 14, 21))
calc_p(c1=c(0, 12, 8, 15), c2=c(1, 21, 14, 21), c3= c(1 , 5, 2, 5), require_c3=T)
calc_p(c1=c(0, 3, 4, 1), c2=c(1, 21, 14, 21), c3= c(1 , 5, 2, 5), require_c3=T)
calc_p(c1=c(1 , 5, 2, 5), c2=c(1, 21, 14, 21))
calc_p(c1=c(1, 21, 14, 21), c2=c(4, 40, 20, 62))

calc_p(c1=c(0, 2, 0, 0), c2=c(2, 31, 14, 32))
calc_p(c1=c(1, 16, 8, 20), c2=c(2, 31, 14, 32), c3= c(1 , 7, 2, 8), require_c3=T)
calc_p(c1=c(0, 6, 4, 4), c2=c(2, 31, 14, 32), c3= c(1 , 7, 2, 8), require_c3=T)
calc_p(c1=c(1 , 7, 2, 8), c2=c(2, 31, 14, 32))
calc_p(c1=c(2, 31, 14, 32), c2=c(4, 40, 20, 62))

calc_p(c1=c(1,4,4,12),)

c1 = c(79, 90, 54, 186)
c2 = c(21, 17, 14, 53)
c3 = c(7, 17, 20, 61)
c4 = c(2, 2, 1, 1)
c5 = c(6, 35, 7, 14)
c6 = c(1, 45, 0, 6)
tab <- rbind(c(70, 88, 62, 208),
             c(46, 116, 34, 113))
chisq.test(tab,)
tab <- rbind(c(67, 94, 52, 189),
             c(49, 110, 44, 132))
chisq.test(tab,)
tab <- rbind(c(31, 57, 25, 59),
             c(85, 147, 71, 262))
chisq.test(tab,)
tab <- rbind(c(42, 59, 31, 83),
             c(90, 194, 84, 304))
chisq.test(tab,)
tab <- rbind(c(4, 40, 20, 62),
             c(111, 164,75, 257))
chisq.test(tab)
fisher.test(tab)
memory.size(size = 4096)
tab <- rbind(c(4, 7, 2, 17),
             c(45, 106, 35, 177),
             c(66, 91, 58, 125))
fisher.test(tab)

calc_p <- function(c){
  c1 <- c(116, 206, 96, 321) - c
  tab <- rbind(c, c1)
  print(chisq.test(tab))
}
calc_pf <- function(c){
  c1 <- c(116, 206, 96, 321) - c
  tab <- rbind(c, c1)
  print(fisher.test(tab))
}
calc_p(c(79, 90, 54,186))
calc_pf(c(21, 17, 14, 53))
calc_p(c(7, 17, 20, 61))
calc_pf(c(2, 2, 1, 1))
calc_p((c(6, 35, 7, 14)))


#molecular subtypes
library(survminer)
library(survival)

df <- read.csv("/Users/huyanshen/Desktop/4-8-table/four-types.csv")
colnames(df)
df$Molecular.Subtypes <- factor(df$Molecular.Subtypes, levels = c("WNT", "SHH", "G3", "G4"))
surv.object<-Surv(df$OS.impute..months., df$Survival.Status)

km=survfit(formula=surv.object ~ Molecular.Subtypes,data = df)

tiff("/Users/huyanshen/Desktop/four-types.tiff", width=17.75, height=17.75, units="cm", res=300)
ggsurvplot(km,break.time.by = 12,xlim = c(0,60),risk.table = TRUE, pval=TRUE, pval.method = TRUE, palette= c('#4F95DB', '#DB2F36','#D6B709', '#187D2A'),legend=c(0.8,0.2),legend.labs=c("WNT", "SHH", "G3", "G4"))
dev.off()

tiff("./km_g_combined_revision_CI.tiff", width=17.75, height=17.75, units="cm", res=300)
ggsurvplot(km,break.time.by = 12,xlim = c(0,60),risk.table = TRUE, pval=TRUE, pval.method = TRUE, palette= c('#4F95DB', '#DB2F36','#D6B709', '#187D2A'),legend=c(0.8,0.2),legend.labs=c("WNT", "SHH", "G3", "G4"), conf.int = TRUE)
dev.off()

#Asia Vs China
df <- read.csv("/Users/huyanshen/Desktop/4-8-table/AsiaVsUs-shh.csv")
colnames(df)
surv.object<-Surv(df$OS.impute..months., df$Survival.Status)
help("ggsurvplot")

km=survfit(formula=surv.object ~ Geographical.Location,data = df)

tiff("/Users/huyanshen/Desktop/km_AsiaUS_revision.tiff", width=17.75, height=17.75, units="cm", res=300)
ggsurvplot(km,break.time.by = 12,xlim = c(0,60),risk.table = TRUE,pval=TRUE, pval.method = TRUE, palette= c('#DB2F36','#4F95DB'),legend=c(0.8,0.2),legend.labs=c("East Asia", "North America"))
dev.off()

tiff("./km_AsiaUS_revision_CI.tiff", width=17.75, height=17.75, units="cm", res=300)
ggsurvplot(km,break.time.by = 12,xlim = c(0,60),risk.table = TRUE,pval=TRUE, pval.method = TRUE, palette= c('#DB2F36','#4F95DB'),legend=c(0.8,0.2),legend.labs=c("East Asia","North America"), conf.int = TRUE)
dev.off()






library(survminer)
library(survival)
library(ggplot2)

df <- read.csv("/Users/huyanshen/Desktop/4-8-table/four-types.csv")
colnames(df)
df$Molecular.Subtypes <- factor(df$Molecular.Subtypes, levels = c("WNT", "SHH", "G3", "G4"))
surv.object<-Surv(df$OS.impute..months., df$Survival.Status)

km=survfit(formula=surv.object ~ Molecular.Subtypes,data = df)

#tiff("./km_preview.tiff", width=17.75, height=17.75, units="cm", res=300)
g <- ggsurvplot(km,
           break.time.by = 12,
           xlim=c(0, 60),
           palette= c('#4F95DB', '#DB2F36','#D6B709', '#187D2A'),
           legend=c(0.2,0.2),
           legend.labs=c("WNT", "SHH", "G3", "G4")) 
g$plot <- g$plot + theme_void()
        
ggsave("survival_plot.png", plot = g$plot, width = 8, height = 6, dpi = 300)

g <- ggsurvplot(km,
                break.time.by = 12,
                xlim=c(0, 60),
                palette= c('#4F95DB', '#DB2F36','#D6B709', '#187D2A'),
                legend=c(0.2,0.2),
                legend.labs=c("WNT", "SHH", "G3", "G4"),
                conf.int = TRUE) 


g$plot <- g$plot + theme_void()
ggsave("survival_plot_CI.png", plot = g$plot, width = 8, height = 6, dpi = 300)

