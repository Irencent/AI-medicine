#######################################
# Author: Fatma Gunturkun
# Date started: 01/25/23
# Last updated: 01/25/23
######################################

# load packages 

library(survival)
library(KMsurv)
library(readxl)
library(readr)
library(dplyr)
library(rms)

#read dataset
df <- read_excel("./Medulloblastoma-Dataset-2023-1-23-share-Fatmat.xlsx")

#remove non-numeric values
df$DFS <- parse_number(df$DFS)
df$OS <- parse_number(df$OS)
df$Sex<-gsub("-",NA,df$Sex)
df$Age<-gsub("-",NA,df$Age)
df$HistologicalSubtypes<-gsub("-",NA,df$HistologicalSubtypes)
df$TumorTexture<-gsub("-",NA,df$TumorTexture)
df$Hydrocephalus<-gsub("-",NA,df$Hydrocephalus)
df$TumorPosition<-gsub("-",NA,df$TumorPosition)

#relabel mortality,age
df <- df %>% mutate(mortality = case_when(Survival_status == "Y" ~ 0,
                                          Survival_status == "N" ~ 1))
df$Age <- findInterval(df$Age, c(6, 18))

#life table, Kaplan Meier, log rank test

#Total
#remove NA
df_t<-subset(df, (!is.na(df[,8])) & (!is.na(df[,9])))
surv.object<-Surv(df_t$OS, df_t$mortality)
km.by<-npsurv(surv.object ~ 1,data = df_t, conf.type = "log-log")
print(km.by)
#5 years survival rates
summary(km.by, times=seq(0,60,60))

survplot(km.by,conf.int = 0.95,
         lty = c(2,1), col = c("black","gray47"), xlab = "Time", ylab = "Survival probability",
         xlim = c(0,120),time.inc = 24, label.curves = T, n.risk = F)


#sex
#remove NA
df_os_sex<-subset(df, (!is.na(df[,8])) & (!is.na(df[,9]))&(!is.na(df[,3])))
surv.object<-Surv(df_os_sex$OS, df_os_sex$mortality)
km.by<-npsurv(surv.object ~ Sex,data = df_os_sex, conf.type = "log-log")
print(km.by)
#5 years survival rates
summary(km.by, times=seq(0,60,60))

survplot(km.by,conf.int = 0.95,
         lty = c(2,1), col = c("black","gray47"), xlab = "Time", ylab = "Survival probability",
         xlim = c(0,120),time.inc = 24, label.curves = T, n.risk = F)
survdiff(surv.object ~ Sex,data = df_os_sex)

#Age
#remove NA
df_os_age<-subset(df, (!is.na(df[,8])) & (!is.na(df[,9]))&(!is.na(df[,2])))
surv.object<-Surv(df_os_age$OS, df_os_age$mortality)
km.by<-npsurv(surv.object ~ Age,data = df_os_age, conf.type = "log-log")
print(km.by)
#5 years survival rates
summary(km.by, times=seq(0,60,60))

survplot(km.by,conf.int = 0.95,
         lty = c(2,1), col = c("black","gray47"), xlab = "Time", ylab = "Survival probability",
         xlim = c(0,120),time.inc = 24, label.curves = T, n.risk = F)
survdiff(surv.object ~ Age,data = df_os_age)

#Tumor resection
#remove NA
df_os<-subset(df, (!is.na(df[,8])) & (!is.na(df[,9])))
surv.object<-Surv(df_os$OS, df_os$mortality)
km.by<-npsurv(surv.object ~ TumorResection,data = df_os, conf.type = "log-log")
print(km.by)
#5 years survival rates
summary(km.by, times=seq(0,60,60))

survplot(km.by,conf.int = 0.95,
         lty = c(2,1), col = c("black","gray47"), xlab = "Time", ylab = "Survival probability",
         xlim = c(0,120),time.inc = 24, label.curves = T, n.risk = F)
survdiff(surv.object ~ TumorResection,data = df_os)

#HistologicalSubtypes
#remove NA
df_os_hs<-subset(df, (!is.na(df[,8])) & (!is.na(df[,9]))&(!is.na(df[,11])))
surv.object<-Surv(df_os_hs$OS, df_os_hs$mortality)
km.by<-npsurv(surv.object ~ HistologicalSubtypes,data = df_os_hs, conf.type = "log-log")
print(km.by)
#5 years survival rates
summary(km.by, times=seq(0,60,60))

survplot(km.by,conf.int = 0.95,
         lty = c(2,1), col = c("black","gray47"), xlab = "Time", ylab = "Survival probability",
         xlim = c(0,120),time.inc = 24, label.curves = T, n.risk = F)
survdiff(surv.object ~ HistologicalSubtypes,data = df_os_hs)

#TumorTexture
#remove NA
df_os_tt<-subset(df, (!is.na(df[,8])) & (!is.na(df[,9]))&(!is.na(df[,13])))
surv.object<-Surv(df_os_tt$OS, df_os_tt$mortality)
km.by<-npsurv(surv.object ~ TumorTexture,data = df_os_tt, conf.type = "log-log")
print(km.by)
#5 years survival rates
summary(km.by, times=seq(0,60,60))

survplot(km.by,conf.int = 0.95,
         lty = c(2,1), col = c("black","gray47"), xlab = "Time", ylab = "Survival probability",
         xlim = c(0,120),time.inc = 24, label.curves = T, n.risk = F)
survdiff(surv.object ~ TumorTexture,data = df_os_tt)

#Hydrocephalus
#remove NA
df_os_hy<-subset(df, (!is.na(df[,8])) & (!is.na(df[,9]))&(!is.na(df[,14])))
surv.object<-Surv(df_os_hy$OS, df_os_hy$mortality)
km.by<-npsurv(surv.object ~ Hydrocephalus,data = df_os_hy, conf.type = "log-log")
print(km.by)
#5 years survival rates
summary(km.by, times=seq(0,60,60))

survplot(km.by,conf.int = 0.95,
         lty = c(2,1), col = c("black","gray47"), xlab = "Time", ylab = "Survival probability",
         xlim = c(0,120),time.inc = 24, label.curves = T, n.risk = F)
survdiff(surv.object ~ Hydrocephalus,data = df_os_hy)

#TumorPosition
#remove NA
df_os_tp<-subset(df, (!is.na(df[,8])) & (!is.na(df[,9]))&(!is.na(df[,15])))
surv.object<-Surv(df_os_tp$OS, df_os_tp$mortality)
km.by<-npsurv(surv.object ~ TumorPosition,data = df_os_tp, conf.type = "log-log")
print(km.by)
#5 years survival rates
summary(km.by, times=seq(0,60,60))

survplot(km.by,conf.int = 0.95,
         lty = c(2,1), col = c("black","gray47"), xlab = "Time", ylab = "Survival probability",
         xlim = c(0,120),time.inc = 24, label.curves = T, n.risk = F)
survdiff(surv.object ~ TumorPosition,data = df_os_tp)

#ssh
df$ssh <- paste(df$MolecularSubtypes,df$GeneticMutations)
df_ssh<-subset(df, (df[,"ssh"]== "SHH TP53-") | (df[,"ssh"]== "SHH TP53+"))
surv.object<-Surv(df_ssh$OS, df_ssh$mortality)
km.by<-npsurv(surv.object ~ ssh,data = df_ssh, conf.type = "log-log")
print(km.by)
#5 years survival rates
summary(km.by, times=seq(0,60,60))

survplot(km.by,conf.int = 0.95,
         lty = c(2,1), col = c("black","gray47"), xlab = "Time", ylab = "Survival probability",
         xlim = c(0,120),time.inc = 24, label.curves = T, n.risk = F)
survdiff(surv.object ~ ssh,data = df_ssh)

#G3/G4
df_g<-subset(df, (df[,"MolecularSubtypes"]== "G3") | (df[,"MolecularSubtypes"]== "G4"))
surv.object<-Surv(df_g$OS, df_g$mortality)
km.by<-npsurv(surv.object ~ MolecularSubtypes,data = df_g, conf.type = "log-log")
print(km.by)
#5 years survival rates
summary(km.by, times=seq(0,60,60))

survplot(km.by,conf.int = 0.95,
         lty = c(2,1), col = c("black","gray47"), xlab = "Time", ylab = "Survival probability",
         xlim = c(0,120),time.inc = 24, label.curves = T, n.risk = F)
survdiff(surv.object ~ MolecularSubtypes,data = df_g)

#coxph model
df_cox<-subset(df, (!is.na(df[,8])) & (!is.na(df[,9]))&(!is.na(df[,3]))&(!is.na(df[,2]))&(!is.na(df[,11]))&(!is.na(df[,13]))&(!is.na(df[,14]))&(!is.na(df[,15])))
surv.object<-Surv(df_cox$OS, df_cox$mortality)
sm<-coxph(surv.object ~ relevel(factor(Sex),ref="m")+relevel(factor(Age),ref="0")+relevel(factor(TumorResection),ref="GTR")
          +relevel(factor(HistologicalSubtypes),ref="1")+relevel(factor(TumorTexture),ref="S")
          +relevel(factor(Hydrocephalus),ref="N")
          +relevel(factor(TumorPosition),ref="1"), data = df_cox)
or<-exp(coef(sm)) # exponentiated coefficients
ci<-exp(confint(sm)) # 95% CI for exponentiated coefficients
cbind(or,ci)
#################################################################################
################################################################################
#comparison of the cohorts
df <- read.csv("./MB_3_cohort-Joyce-update.csv", stringsAsFactors=FALSE, na.strings = "" )
factorVars1 <- c("Sex","Race_1","MolecularSubtypes_1","Survival_status","Recurrence-label","HistologicalSubtypes","TumorResection","TumorTexture","Hydrocephalus","TumorPosition")
vars1 <- c("Age","Sex","Race_1","MolecularSubtypes_1","Survival_status","Recurrence-label","HistologicalSubtypes","TumorResection","TumorTexture","Hydrocephalus","TumorPosition")
table1 <- CreateTableOne(vars = vars1, strata = c("Cohort"), data = df, factorVars = factorVars1)
table1 <- print(table1,smd = TRUE)
write.csv(table1, "./table1.csv")

df <- df[!is.na(df$OS_missing), ]
df <- df[!is.na(df$Survival_status), ]

surv.object<-Surv(df$OS_missing, df$Survival_status)
km.by<-npsurv(surv.object ~ Cohort,data = df, conf.type = "log-log")
print(km.by)
#5 years survival rates
summary(km.by, times=seq(0,60,60))

survplot(km.by,conf.int = 0.95,
         lty = c(2,1), col = c("black","gray47"), xlab = "Time", ylab = "Survival probability",
         xlim = c(0,120),time.inc = 24, label.curves = T, n.risk = F)
survdiff(surv.object ~ Cohort,data = df)

###############################################
#event times in 3 cohorts
df_death<-subset(df, df[,"Survival_status"]== 1)
table2 <- CreateTableOne(vars = ("OS_missing"), strata = c("Cohort"), data = df_death)
table2 <- print(table2,smd = TRUE)

################################################################################
#missing_data_imputation
#read dataset
library(mice)
install.packages( "/Users/faydinlik/Desktop/projects/quan_zhou/DMwR_0.4.1.tar.gz", repos=NULL, type="source" )
install.packages("ROCR")
library(DMwR)
df_1<-subset(df, df[,"Cohort"]== "1")
impute.index  <- df_1$index_os
impute.index  <- impute.index[!is.na(impute.index)]
df_1$OperateDate <- as.Date(df_1$OperateDate)
df_1$time <- difftime(df_1$OperateDate,min(df_1$OperateDate), units = "days")/30
df_1 <- df_1[,c("Age","Sex","time","MolecularSubtypes_1","Survival_status","OS_missing","Recurrence_label","HistologicalSubtypes","DFS_missing","TumorResection","TumorTexture","Hydrocephalus","TumorPosition")]
train.impute <- df_1
train.impute$Sex<-factor(train.impute$Sex)
train.impute$MolecularSubtypes_1<-factor(train.impute$MolecularSubtypes_1)
train.impute$Survival_status<-factor(train.impute$Survival_status)
train.impute$Recurrence_label<-factor(train.impute$Recurrence_label)
train.impute$HistologicalSubtypes<-factor(train.impute$HistologicalSubtypes)
train.impute$TumorResection<-factor(train.impute$TumorResection)
train.impute$TumorTexture<-factor(train.impute$TumorTexture)
train.impute$Hydrocephalus<-factor(train.impute$Hydrocephalus)
train.impute$TumorPosition<-factor(train.impute$TumorPosition)
train.impute$OS_missing[impute.index] <- NA
train.mice <- train.impute

impmethods <- c("pmm","sample","rf","cart")
for (i in 1:length(impmethods)){
  print(paste0(impmethods[i], ":"))
  miceimp <- mice(data = train.mice, m = 1,maxit = 10, method = impmethods[i], printFlag = F)
  micepred<- complete(miceimp)
  print(regr.eval(micepred$OS_missing[impute.index],df_1$OS_missing[impute.index]))
}

train.imputed <- df_1
train.imputed$Sex<-factor(train.imputed$Sex)
train.imputed$MolecularSubtypes_1<-factor(train.imputed$MolecularSubtypes_1)
train.imputed$Survival_status<-factor(train.imputed$Survival_status)
train.imputed$Recurrence_label<-factor(train.imputed$Recurrence_label)
train.imputed$HistologicalSubtypes<-factor(train.imputed$HistologicalSubtypes)
train.imputed$TumorResection<-factor(train.imputed$TumorResection)
train.imputed$TumorTexture<-factor(train.imputed$TumorTexture)
train.imputed$Hydrocephalus<-factor(train.imputed$Hydrocephalus)
train.imputed$TumorPosition<-factor(train.imputed$TumorPosition)

miceimp <- mice(data = train.imputed, m = 1,maxit = 10, method = "cart", printFlag = F)
complete1 <- complete(miceimp,1)
write.csv(complete1,'complete1.csv')

#combat
library(devtools)
install_github("jfortin1/neuroCombat_Rpackage")
library(neuroCombat)
#read dataset
df <- read.csv("./MB_3_cohort-Joyce-update.csv", stringsAsFactors=FALSE, na.strings = "" )
df <- df %>% select(where(~ n_distinct(.) > 1))
dim(df)
dat=t(df[,58:6899])
dim(dat)

#parametric combat
data.harmonized <- neuroCombat(dat=dat, batch=df$batch)
dim(data.harmonized$dat.combat)
write.csv(t(data.harmonized$dat.combat),'combat_parametric.csv')

#nonparametric combat
data.harmonized <- neuroCombat(dat=dat, batch=df$batch, parametric=FALSE)
write.csv(t(data.harmonized$dat.combat),'combat_nonparametric.csv')

#adjusted combat
mod <- model.matrix(~df$Age+df$Sex)
combat.harmonized <- neuroCombat(dat=dat, batch=df$batch, mod=mod)
write.csv(t(combat.harmonized$dat.combat),'combat_harmonized.csv')

############################################################
#molecular subtypes
################################################################################
#missing_data_imputation
#read dataset
library(mice)
install.packages( "/Users/faydinlik/Desktop/projects/quan_zhou/DMwR_0.4.1.tar.gz", repos=NULL, type="source" )
install.packages("ROCR")
library(DMwR)
install.packages("anytime")   # Install anytime package
library("anytime") 

setwd("/Users/faydinlik/Library/CloudStorage/Box-Box/MB study - QSU/JAMA-submission/fatma_analysis")
df_1 <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
impute.index  <- df_1$index_os
impute.index  <- impute.index[!is.na(impute.index)]

colnames(df_1)
df_1 <- df_1[,c("Age","Sex","mol_sub","Survival.status","OS_missing","Recurrence.label_missing","HistologicalSubtypes","DFS_missing","TumorTexture","Hydrocephalus","TumorPosition")]
train.impute <- df_1
train.impute$Sex<-factor(train.impute$Sex)
train.impute$mol_sub<-factor(train.impute$mol_sub)
train.impute$Survival.status<-factor(train.impute$Survival.status)
train.impute$Recurrence.label_missing<-factor(train.impute$Recurrence.label_missing)
train.impute$HistologicalSubtypes<-factor(train.impute$HistologicalSubtypes)
train.impute$TumorTexture<-factor(train.impute$TumorTexture)
train.impute$Hydrocephalus<-factor(train.impute$Hydrocephalus)
train.impute$TumorPosition<-factor(train.impute$TumorPosition)
train.impute$OS_missing[impute.index] <- NA
train.mice <- train.impute

impmethods <- c("pmm","sample","rf","cart")
for (i in 1:length(impmethods)){
  print(paste0(impmethods[i], ":"))
  miceimp <- mice(data = train.mice, m = 1,maxit = 10, method = impmethods[i], printFlag = F)
  micepred<- complete(miceimp)
  print(regr.eval(micepred$OS_missing[impute.index],df_1$OS_missing[impute.index]))
}

train.imputed <- df_1
train.imputed$Sex<-factor(train.imputed$Sex)
train.imputed$mol_sub<-factor(train.imputed$mol_sub)
train.imputed$Survival.status<-factor(train.imputed$Survival.status)
train.imputed$Recurrence.label_missing<-factor(train.imputed$Recurrence.label_missing)
train.imputed$HistologicalSubtypes<-factor(train.imputed$HistologicalSubtypes)
train.imputed$TumorTexture<-factor(train.imputed$TumorTexture)
train.imputed$Hydrocephalus<-factor(train.imputed$Hydrocephalus)
train.imputed$TumorPosition<-factor(train.imputed$TumorPosition)

miceimp <- mice(data = train.imputed, m = 1,maxit = 10, method = "cart", printFlag = F)
complete1 <- complete(miceimp,1)
write.csv(complete1,'complete1.csv')

#############################################
#survival for molecular subtypes

df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
#df<-subset(df, df[,"mol_sub"]!= "nonWNT/nonSHHno")

#relabel age
df$Age_nomissing <- findInterval(df$Age_nomissing, c(6,18))
df$OS_nomissing_adjusted <- as.numeric(df$OS_nomissing_adjusted)
df$mol_sub <- as.factor(df$mol_sub)

#coxph model
surv.object<-Surv(df$OS_nomissing_adjusted, df$mortality)
'''
sm<-coxph(Surv(time, com_event) ~ relevel(factor(Sex),ref="0")+relevel(factor(Age_nomissing),ref="0")+relevel(factor(TumorResection.clincial.reports_nomissing1),ref="GTR")
          +relevel(factor(HistologicalSubtypes_nomissing),ref="1")+relevel(factor(TumorTexture_nomissing),ref="S")
          +relevel(factor(Hydrocephalus),ref="N")+relevel(factor(TumorPosition),ref="1")
          +mol_sub
         , data = df)
'''
sm<-coxph(surv.object ~ relevel(factor(Sex),ref="0")+relevel(factor(Age_nomissing),ref="0")+
          +relevel(factor(HistologicalSubtypes_nomissing),ref="1")+relevel(factor(TumorTexture_nomissing),ref="Soft")
          +relevel(factor(Hydrocephalus_nomissing),ref="N")+relevel(factor(TumorPosition_nomissing),ref="1")
          +mol_sub
          , data = df)
or<-exp(coef(sm)) # exponentiated coefficients
ci<-exp(confint(sm)) # 95% CI for exponentiated coefficients
cbind(or,ci)

#ph assumption check
cox.zph(sm,global = TRUE)

install.packages('multcomp')
library('multcomp')
### specify all pair-wise comparisons among levels of "mol_sub"
sm.glht <- glht(sm, mcp(mol_sub = "Tukey"))
summary(sm.glht)
or<-exp(coef(sm.glht)) # exponentiated coefficients
ci<-exp(confint(sm.glht)) # 95% CI for exponentiated coefficients
cbind(or,ci)

#life table, Kaplan Meier, log rank test

#ssh
install.packages("survival")
library(survival)
df_ssh<-subset(df, (df[,"mol_sub"]== "SHHTP53-") | (df[,"mol_sub"]== "SHHTP53+"))
surv.object<-Surv(df_ssh$OS_nomissing_adjusted, df_ssh$mortality)
kmfit = survfit(surv.object ~ df_ssh$mol_sub)
plot(kmfit, lty = c("solid", "dashed"), col = c("black", "grey"), xlab = "Survival Time In Months", ylab = "Survival Probabilities")
legend("topright", c("SHHTP53-", "SHHTP53+"), lty = c("solid", "dashed"), col = c("black", "grey"))
survdiff(surv.object ~ mol_sub,data = df_ssh)
km.by<-npsurv(surv.object ~ mol_sub,data = df_ssh, conf.type = "log-log")
print(km.by)
#5 years survival rates
summary(km.by, times=seq(0,60,60))
#G3/G4
df_g<-subset(df, (df[,"mol_sub"]== "G3") | (df[,"mol_sub"]== "G4"))
surv.object<-Surv(df_g$OS_nomissing_adjusted, df_g$mortality)
kmfit = survfit(surv.object ~ df_g$mol_sub)
plot(kmfit, lty = c("solid", "dashed"), col = c("black", "grey"), xlab = "Survival Time In Months", ylab = "Survival Probabilities")

legend("topright", c("G3", "G4"), lty = c("solid", "dashed"), col = c("black", "grey"))
survdiff(surv.object ~ mol_sub,data = df_g)
km.by<-npsurv(surv.object ~ mol_sub,data = df_g, conf.type = "log-log")
print(km.by)
#5 years survival rates
summary(km.by, times=seq(0,60,60))

## 4 subtypes(G3/G4, SHH_TP53+, SHH_TP53-, WNT)
data1 = subset(df, (df[,"mol_sub"]!= "nonWNT/nonSHH") & (df[,"mol_sub"]!= "SHH"))
data1$subtype = as.character(data1$mol_sub)
data1$subtype[which(data1$subtype=="G3")] = "G3/G4"
data1$subtype[which(data1$subtype=="G4")] = "G3/G4"
data1$subtype = as.factor(data1$subtype)
summary(data1$subtype)


#survC1 C index
install.packages('survC1')
library(survC1)
df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
df$Age_nomissing<- findInterval(df$Age_nomissing, c(3.1))
N = 1000
C_index3 = rep(0, N)
C_index4 = rep(0, N)
n = nrow(data1)
for (i in 1:N){ 
  if(i %% 200 == 0) cat(i, "times of trying...\n")
  # redevide the train and test data in each epoch
  set.seed(i)
  index.training=sample(n, 0.75 * n, F)
  index.testing=(1:n)[-index.training]
  
  res.cox <- coxph(Surv(OS_nomissing_adjusted, mortality) ~ relevel(factor(Sex),ref="0")+relevel(factor(Age_nomissing),ref="0")+
                     +relevel(factor(HistologicalSubtypes_nomissing),ref="1")+relevel(factor(TumorTexture_nomissing),ref="Soft")
                   +relevel(factor(Hydrocephalus_nomissing),ref="N")+relevel(factor(TumorPosition_nomissing),ref="1")+subtype, data = data1[index.training, ])
  # test on the test_data
  marker <- as.vector(predict(res.cox, data1[index.testing, ], type = 'lp'))
  
  # tau = max(data$OS[index.testing])
  tau=quantile(data1$OS_nomissing_adjusted[index.testing], 0.95)
  D=cbind(data1$OS_nomissing_adjusted[index.testing], data1$mortality[index.testing], marker)
  C_index3[i] = max(Inf.Cval(D, tau, itr=200)$Dhat, 1-Inf.Cval(D, tau, itr=200)$Dhat)
  C_index4[i] = Inf.Cval(D, tau, itr=200)$Dhat
}
print(C_index3[1:10])
print(C_index4[1:10])
cat("The mean of survC1 C index in", N, "epochs =", mean(C_index3))



#Harrell C index

C_index1 = rep(0, N)
C_index2 = rep(0, N)
for (i in 1:N){
  if(i %% 200 == 0) cat(i, "times of trying...\n")
  
  set.seed(i)
  index.training=sample(n, 0.75 * n, F)
  index.testing=(1:n)[-index.training]
  
  res.cox <- coxph(Surv(OS_nomissing_adjusted, mortality) ~ relevel(factor(Sex),ref="0")+relevel(factor(Age_nomissing),ref="0")+
                     +relevel(factor(HistologicalSubtypes_nomissing),ref="1")+relevel(factor(TumorTexture_nomissing),ref="Soft")
                   +relevel(factor(Hydrocephalus_nomissing),ref="N")+relevel(factor(TumorPosition_nomissing),ref="1")+subtype, data = data1[index.training, ])
  # C_index[i] = survConcordance(Surv(data[index.testing,"OS"],data[index.testing,"censored"])~predict(res.cox,data[index.training,]))$concordance
  index_i = concordancefit(Surv(data1[index.testing,"OS_nomissing_adjusted"],data1[index.testing,"mortality"]),predict(res.cox,data1[index.testing,]))$concordance
  C_index1[i] = max(index_i, 1-index_i)
  C_index2[i] = index_i
}
print(C_index1[1:10])
print(C_index2[1:10])
cat("The mean of Harrell C index in", N, "epochs =", mean(C_index1))
```

## 4 subtypes(G3, G4, SHH, WNT)

data2 = subset(df, (df[,"mol_sub"]!= "nonWNT/nonSHH") & (df[,"mol_sub"]!= "SHH"))
data2$subtype = as.character(data2$mol_sub)
data2$subtype[which(data2$subtype=="SHHTP53+")] = "SHH"
data2$subtype[which(data2$subtype=="SHHTP53-")] = "SHH"
data2$subtype = as.factor(data2$subtype)
summary(data2$subtype)
```

#survC1 C index

N = 1000
C_index5 = rep(0, N)
n = nrow(data2)
for (i in 1:N){
  if(i %% 200 == 0) cat(i, "times of trying...\n")
  # redevide the train and test data in each epoch
  set.seed(i)
  index.training=sample(n, 0.75 * n, F)
  index.testing=(1:n)[-index.training]
  
  res.cox <-coxph(Surv(OS_nomissing_adjusted, mortality) ~ relevel(factor(Sex),ref="0")+relevel(factor(Age_nomissing),ref="0")+
          +relevel(factor(HistologicalSubtypes_nomissing),ref="1")+relevel(factor(TumorTexture_nomissing),ref="Soft")
        +relevel(factor(Hydrocephalus_nomissing),ref="N")+relevel(factor(TumorPosition_nomissing),ref="1")+subtype, data = data2[index.training, ])
  # test on the test_data
  marker <- as.vector(predict(res.cox, data2[index.testing, ], type = 'lp'))
  
  # tau = max(data$OS[index.testing])
  tau=quantile(data2$OS_nomissing_adjusted[index.testing], 0.95)
  D=cbind(data2$OS_nomissing_adjusted[index.testing], data2$mortality[index.testing], marker)
  C_index5[i] = max(Inf.Cval(D, tau, itr=200)$Dhat, 1-Inf.Cval(D, tau, itr=200)$Dhat)
}
print(C_index5[1:10])
cat("The mean of survC1 C index in", N, "epochs =", mean(C_index5))
```

#Harrell C index

C_index6 = rep(0, N)
for (i in 1:N){
  if(i %% 200 == 0) cat(i, "times of trying...\n")
  
  set.seed(i)
  index.training=sample(n, 0.75 * n, F)
  index.testing=(1:n)[-index.training]
  
  res.cox <-coxph(Surv(OS_nomissing_adjusted, mortality) ~ relevel(factor(Sex),ref="0")+relevel(factor(Age_nomissing),ref="0")+
                    +relevel(factor(HistologicalSubtypes_nomissing),ref="1")+relevel(factor(TumorTexture_nomissing),ref="Soft")
                  +relevel(factor(Hydrocephalus_nomissing),ref="N")+relevel(factor(TumorPosition_nomissing),ref="1")+subtype, data = data2[index.training, ])
  # test on the test_data
  # C_index[i] = survConcordance(Surv(data[index.testing,"OS"],data[index.testing,"censored"])~predict(res.cox,data[index.training,]))$concordance
  index_i = concordancefit(Surv(data2[index.testing,"OS_nomissing_adjusted"],data2[index.testing,"mortality"]),predict(res.cox,data2[index.testing,]))$concordance
  C_index6[i] = max(index_i, 1-index_i)
}
print(C_index6[1:10])
cat("The mean of Harrell C index in", N, "epochs =", mean(C_index6))
```
quantile(C_index3, probs = c(.025,0.5, .975))
quantile(C_index1, probs = c(.025,0.5,  .975))
quantile(C_index5, probs = c(.025,0.5,  .975))
quantile(C_index6, probs = c(.025,0.5,  .975))

###############################################################################
## 4 subtypes(G3/G4, SHH_TP53+, SHH_TP53-, WNT)
data1 = subset(df, (df[,"mol_sub"]!= "nonWNT/nonSHH") & (df[,"mol_sub"]!= "SHH"))
data1$subtype = as.character(data1$mol_sub)
data1$subtype[which(data1$subtype=="G3")] = "G3/G4"
data1$subtype[which(data1$subtype=="G4")] = "G3/G4"
data1$subtype = as.factor(data1$subtype)
summary(data1$subtype)


#survC1 C index
install.packages('survC1')
library(survC1)
N = 1000
C_index3 = rep(0, N)
C_index4 = rep(0, N)
n = nrow(data1)
for (i in 1:N){ 
  if(i %% 200 == 0) cat(i, "times of trying...\n")
  # redevide the train and test data in each epoch
  set.seed(i)
  index.training=sample(n, 0.75 * n, F)
  index.testing=(1:n)[-index.training]
  
  res.cox <- coxph(Surv(OS_nomissing_adjusted, mortality) ~ subtype, data = data1[index.training, ])
  # test on the test_data
  marker <- as.vector(predict(res.cox, data1[index.testing, ], type = 'lp'))
  
  # tau = max(data$OS[index.testing])
  tau=quantile(data1$OS_nomissing_adjusted[index.testing], 0.95)
  D=cbind(data1$OS_nomissing_adjusted[index.testing], data1$mortality[index.testing], marker)
  C_index3[i] = max(Inf.Cval(D, tau, itr=200)$Dhat, 1-Inf.Cval(D, tau, itr=200)$Dhat)
  C_index4[i] = Inf.Cval(D, tau, itr=200)$Dhat
}
print(C_index3[1:10])

cat("The mean of survC1 C index in", N, "epochs =", mean(C_index3))



Harrell C index

C_index1 = rep(0, N)
C_index2 = rep(0, N)
for (i in 1:N){
  if(i %% 200 == 0) cat(i, "times of trying...\n")
  
  set.seed(i)
  index.training=sample(n, 0.75 * n, F)
  index.testing=(1:n)[-index.training]
  
  res.cox <- coxph(Surv(OS_nomissing_adjusted, mortality) ~ subtype, data = data1[index.training, ])
  # C_index[i] = survConcordance(Surv(data[index.testing,"OS"],data[index.testing,"censored"])~predict(res.cox,data[index.training,]))$concordance
  index_i = concordancefit(Surv(data1[index.testing,"OS_nomissing_adjusted"],data1[index.testing,"mortality"]),predict(res.cox,data1[index.testing,]))$concordance
  #C_index1[i] = max(index_i, 1-index_i)
  C_index2[i] = index_i
}
#print(C_index1[1:10])
print(C_index2[1:10])
cat("The mean of Harrell C index in", N, "epochs =", mean(C_index2))
```

## 4 subtypes(G3, G4, SHH, WNT)

data2 = subset(df, (df[,"mol_sub"]!= "nonWNT/nonSHH") & (df[,"mol_sub"]!= "SHH"))
data2$subtype = as.character(data2$mol_sub)
data2$subtype[which(data2$subtype=="SHHTP53+")] = "SHH"
data2$subtype[which(data2$subtype=="SHHTP53-")] = "SHH"
data2$subtype = as.factor(data2$subtype)
summary(data2$subtype)



#survC1 C index

N = 1000
C_index5 = rep(0, N)
n = nrow(data2)
for (i in 1:N){
  if(i %% 200 == 0) cat(i, "times of trying...\n")
  # redevide the train and test data in each epoch
  set.seed(i)
  index.training=sample(n, 0.75 * n, F)
  index.testing=(1:n)[-index.training]
  
  res.cox <-coxph(Surv(OS_nomissing_adjusted, mortality) ~ subtype, data = data2[index.training, ])
  # test on the test_data
  marker <- as.vector(predict(res.cox, data2[index.testing, ], type = 'lp'))
  
  # tau = max(data$OS[index.testing])
  tau=quantile(data2$OS_nomissing_adjusted[index.testing], 0.95)
  D=cbind(data2$OS_nomissing_adjusted[index.testing], data2$mortality[index.testing], marker)
  C_index5[i] = max(Inf.Cval(D, tau, itr=200)$Dhat, 1-Inf.Cval(D, tau, itr=200)$Dhat)
}
print(C_index5[1:10])
cat("The mean of survC1 C index in", N, "epochs =", mean(C_index5))
```

#Harrell C index

C_index6 = rep(0, N)
for (i in 1:N){
  if(i %% 200 == 0) cat(i, "times of trying...\n")
  
  set.seed(i)
  index.training=sample(n, 0.75 * n, F)
  index.testing=(1:n)[-index.training]
  
  res.cox <-coxph(Surv(OS_nomissing_adjusted, mortality) ~ subtype, data = data2[index.training, ])
  # test on the test_data
  # C_index[i] = survConcordance(Surv(data[index.testing,"OS"],data[index.testing,"censored"])~predict(res.cox,data[index.training,]))$concordance
  index_i = concordancefit(Surv(data2[index.testing,"OS_nomissing_adjusted"],data2[index.testing,"mortality"]),predict(res.cox,data2[index.testing,]))$concordance
  C_index6[i] = max(index_i, 1-index_i)
}
print(C_index6[1:10])
cat("The mean of Harrell C index in", N, "epochs =", mean(C_index6))
```
quantile(C_index3, probs = c(.025,0.5, .975))
quantile(C_index2, probs = c(.025,0.5, .975))
quantile(C_index5, probs = c(.025,0.5, .975))
quantile(C_index6, probs = c(.025,0.5, .975))

####################################################################
#forest plot
df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
df<-subset(df, df[,"mol_sub"]== "G3"|df[,"mol_sub"]== "G4")
sm<-coxph(Surv(df$OS_nomissing_adjusted, df$mortality) ~ relevel(factor(mol_sub),ref="G4"), data = df)
or<-exp(coef(sm)) # exponentiated coefficients
ci<-exp(confint(sm)) # 95% CI for exponentiated coefficients
ci
cbind(or,ci)

df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
df<-subset(df, df[,"mol_sub"]== "SHHTP53-"|df[,"mol_sub"]== "SHHTP53+")
sm<-coxph(Surv(df$OS_nomissing_adjusted, df$mortality) ~ relevel(factor(mol_sub),ref="SHHTP53-"), data = df)
sm
or<-exp(coef(sm)) # exponentiated coefficients
ci<-exp(confint(sm)) # 95% CI for exponentiated coefficients
cbind(or,ci)

df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
df$Age<- findInterval(df$Age, c(3.1))
sm<-coxph(Surv(df$OS_nomissing_adjusted, df$mortality) ~ relevel(factor(Age),ref="1"), data = df)
or<-exp(coef(sm)) # exponentiated coefficients
ci<-exp(confint(sm)) # 95% CI for exponentiated coefficients
cbind(or,ci)

df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
sm<-coxph(Surv(df$OS_nomissing_adjusted, df$mortality) ~ relevel(factor(Sex),ref="0"), data = df)
or<-exp(coef(sm)) # exponentiated coefficients
ci<-exp(confint(sm)) # 95% CI for exponentiated coefficients
cbind(or,ci)

df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
sm<-coxph(Surv(df$OS_nomissing_adjusted, df$mortality) ~ Hydrocephalus, data = df)
sm
or<-exp(coef(sm)) # exponentiated coefficients
ci<-exp(confint(sm)) # 95% CI for exponentiated coefficients
cbind(or,ci)

df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
sm<-coxph(Surv(df$OS_nomissing_adjusted, df$mortality) ~ HistologicalSubtypes_2, data = df)
or<-exp(coef(sm)) # exponentiated coefficients
ci<-exp(confint(sm)) # 95% CI for exponentiated coefficients
cbind(or,ci)

df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
sm<-coxph(Surv(df$OS_nomissing_adjusted, df$mortality) ~ TumorPosition_2, data = df)
or<-exp(coef(sm)) # exponentiated coefficients
ci<-exp(confint(sm)) # 95% CI for exponentiated coefficients
cbind(or,ci)

df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
sm<-coxph(Surv(df$OS_nomissing_adjusted, df$mortality) ~ relevel(factor(TumorTexture_2),ref="Soft"), data = df)
or<-exp(coef(sm)) # exponentiated coefficients
ci<-exp(confint(sm)) # 95% CI for exponentiated coefficients
cbind(or,ci)

df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
df<-subset(df, df[,"mol_sub"]== "G3"|df[,"mol_sub"]== "G4")
df$Age_nomissing1 <- findInterval(df$Age_nomissing, c(3.1))
sm<-coxph(Surv(df$OS_nomissing_adjusted, df$mortality) ~ relevel(factor(Age_nomissing1),ref="1")+HistologicalSubtypes_nomissing_2+Hydrocephalus_nomissing+relevel(factor(mol_sub),ref="G4")+relevel(factor(Sex),ref="0"), data = df)
or<-exp(coef(sm)) # exponentiated coefficients
ci<-exp(confint(sm)) # 95% CI for exponentiated coefficients
cbind(or,ci)

df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
df<-subset(df, df[,"mol_sub"]== "SHHTP53-"|df[,"mol_sub"]== "SHHTP53+")
df$Age_nomissing1 <- findInterval(df$Age_nomissing, c(3.1))
sm<-coxph(Surv(df$OS_nomissing_adjusted, df$mortality) ~ relevel(factor(Age_nomissing1),ref="1")+HistologicalSubtypes_nomissing_2+Hydrocephalus_nomissing+relevel(factor(mol_sub),ref="SHHTP53-"), data = df)
or<-exp(coef(sm)) # exponentiated coefficients
ci<-exp(confint(sm)) # 95% CI for exponentiated coefficients
cbind(or,ci)

df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
df<-subset(df, df[,"mol_sub"]== "SHHTP53-"|df[,"mol_sub"]== "SHHTP53+")
nrow(df)
df<-subset(df, df[,"Age_nomissing"]>= 4&df[,"Age_nomissing"]<= 16)
#df$Age_nomissing1 <- findInterval(df$Age_nomissing, c(3.1)
sm<-coxph(Surv(df$OS_nomissing_adjusted, df$mortality) ~ HistologicalSubtypes_nomissing_2+Hydrocephalus_nomissing+relevel(factor(mol_sub),ref="SHHTP53-"), data = df)
sm
or<-exp(coef(sm)) # exponentiated coefficients
ci<-exp(confint(sm)) # 95% CI for exponentiated coefficients
cbind(or,ci)
table(df$mol_sub)
table(df$Hydrocephalus_nomissing)
table(df$HistologicalSubtypes_nomissing_2)


install.packages("forestploter")
library(forestploter)

df <- read.csv("./forest_plot_cox_univariate.csv")
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
tiff("forest_univariate.tiff", width=17.75, height=17.75, units="cm", res=300)
p <- forest(df[,c(1:2,8:9,6)],
            est = df$est,
            lower = df$low, 
            upper = df$high,
            ci_column = 3,
            ref_line = 1,
            arrow_lab = c("", ""),
            xlim = c(0,5.5),
            ticks_at = c(0.5,1,2,3,4,5),
            footnote = "")

# Print plot
plot(p)

dev.off()

##################
#forest multivariate
df <- read.csv("./forest_plot_cox_multivariate.csv")
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
tiff("forest_multivariate.tiff", width=19, height=11, units="cm", res=300)
p <- forest(df[,c(1:2,8:9,6)],
            est = df$est,
            lower = df$low, 
            upper = df$high,
            ci_column = 3,
            ref_line = 1,
            arrow_lab = c("", ""),
            xlim = c(0,5.5),
            ticks_at = c(0.5,1,2,3,4,5),
            footnote = "")

# Print plot
plot(p)

dev.off()

##################
#forest multivariate shh
df <- read.csv("./forest_plot_cox_multivariate_shh.csv")
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
tiff("forest_multivariate_shh.tiff", width=19, height=11, units="cm", res=300)
p <- forest(df[,c(1:2,8:9,6)],
            est = df$est,
            lower = df$low, 
            upper = df$high,
            ci_column = 3,
            ref_line = 1,
            arrow_lab = c("", ""),
            xlim = c(0,5.5),
            ticks_at = c(0.5, 1, 2,3,4,5),
            footnote = "")

plot(p)

dev.off()
##################
#forest multivariate shh revision
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
tiff("forest_multivariate_shh_revision.tiff", width=19, height=11, units="cm", res=300)
p <- forest(df[,c(1:2,8:9,6)],
            est = df$est,
            lower = df$low, 
            upper = df$high,
            ci_column = 3,
            ref_line = 1,
            arrow_lab = c("", ""),
            xlim = c(0,5.5),
            ticks_at = c(0.5, 1, 2,3,4,5),
            footnote = "")


# Print plot
plot(p)

dev.off()

#molecular subtypes
library(survminer)

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

km.by<-npsurv(surv.object ~ mol_sub_2,data = df, conf.type = "log-log")
survplot(km.by,conf.int = 0.95,
         lty = c(2,1), col = c("black","gray47"), xlab = "Time", ylab = "Survival probability",
         xlim = c(0,120),time.inc = 24, label.curves = T, n.risk = F)

#################################################
#matrixplot
library(plyr)
library(grid)
df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
df<-subset(df, df[,"mol_sub"]== "SHHTP53-"|df[,"mol_sub"]== "SHHTP53+"|df[,"mol_sub"]== "G3"|df[,"mol_sub"]== "G4"|df[,"mol_sub"]== "WNT")
df$fvi <- ifelse(df$TumorPosition==3,2,df$TumorPosition)
df$Race <- ifelse(df$Race=="More Than One Race"|df$Race=="Native Hawaiian or Other Pacific Islander"|df$Race=="Black or African American","Other",df$Race)
#df$Race <- ifelse(is.na(df$Race),"Missing",df$Race)
df$Age <- findInterval(df$Age, c(3.1,18.1))
df$HistologicalSubtypes <-factor(df$HistologicalSubtypes)
df$Age <-factor(df$Age)
df$fvi <-factor(df$fvi)
df <-arrange(df, mol_sub,HistologicalSubtypes,Age,Hydrocephalus,TumorTexture,fvi,Race) 

row.plot1 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('mol', id, fill = df$mol_sub)) +
  geom_raster(alpha=0.8) +
  scale_fill_manual(name = "Molecular subgroup",
                    values = c('steelblue', 'tomato3','goldenrod','forestgreen','firebrick'),
                    labels = c("G3", "G4",'SHHTP53+','SHHTP53-','WNT')) +
  theme(legend.position=c(-0.026, 0.5),
        legend.title = element_text(size = 5),
        legend.text = element_text(size = 5),
        legend.key.size = unit(0.5, 'cm'), #change legend key size
        legend.key.height = unit(0.4, 'cm'), #change legend key height
        legend.key.width = unit(0.5, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.5,-0.5,-0.5,1), "cm"))+
  labs(x = "",
       y = "", title = "") +
  #scale_fill_discrete(name = "Molecular subgroup")+
  coord_flip()

row.plot2 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('his', id, fill = df$HistologicalSubtypes)) +
  geom_raster(alpha=0.7) +
  theme(legend.position=c(-0.023, 0.5),
        legend.title = element_text(size = 5),
        legend.text = element_text(size = 5),
        legend.key.size = unit(0.5, 'cm'), #change legend key size
        legend.key.height = unit(0.4, 'cm'), #change legend key height
        legend.key.width = unit(0.5, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.5,-0.5,-0.5,1), "cm"))+
  labs(x = "",
       y = "", title = "") +
  scale_fill_manual(name = "Histological subgroup",labels = c('Classic', 'DN', 'MBEN', 'LCA','Missing'),values=c("lightgreen", "#F0E442", "#0072B2", "red"))+
  coord_flip()

row.plot3 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('Age', id, fill = df$Age)) +
  geom_raster(alpha=0.7) +
  scale_fill_manual(name = "Age",
                    values = c('lightcyan1', 'skyblue','blue4'),
                    labels = c("<=3","3-18", ">18","Missing")) +
  theme(legend.position=c(-0.035, 0.5),
        legend.title = element_text(size = 5),
        legend.text = element_text(size = 5),
        legend.key.size = unit(0.5, 'cm'), #change legend key size
        legend.key.height = unit(0.5, 'cm'), #change legend key height
        legend.key.width = unit(0.5, 'cm'),
    axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
    panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
    panel.background = element_blank(),
        plot.margin=unit(c(-0.5,-0.5,-0.5,1), "cm")
  )+
  labs(x = "",
       y = "", title = "") +
  coord_flip()

row.plot4 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('Hyd', id, fill = df$Hydrocephalus)) +
  geom_raster(alpha=0.7) +
  scale_fill_manual(name = "Hydrocephalus",
                    values = c( 'cornflowerblue','firebrick'),
                    labels = c("No","Yes")) +
  theme(legend.position=c(-0.035, 0.5),
        legend.title = element_text(size = 5),
        legend.text = element_text(size = 5),
        legend.key.size = unit(0.5, 'cm'), #change legend key size
        legend.key.height = unit(0.5, 'cm'), #change legend key height
        legend.key.width = unit(0.5, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.5,-0.5,-0.5,1), "cm")
  )+
  labs(x = "",
       y = "", title = "") +
  coord_flip()

row.plot5 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('texture', id, fill = df$TumorTexture)) +
  geom_raster(alpha=0.8) +
  scale_fill_manual(name = "Tumor texture",
                    values = c('purple4', 'violet','thistle1'),
                    labels = c('Hard', "Mix","Soft",'Missing')) +
  theme(legend.position=c(-0.035, 0.5),
        legend.title = element_text(size = 5),
        legend.text = element_text(size = 5),
        legend.key.size = unit(0.5, 'cm'), #change legend key size
        legend.key.height = unit(0.5, 'cm'), #change legend key height
        legend.key.width = unit(0.5, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.5,-0.5,-0.5,1), "cm"))+
  labs(x = "",
       y = "", title = "") +
  #scale_fill_discrete(name = "Molecular subtypes")+
  coord_flip()

row.plot6 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('fvi', id, fill = df$fvi)) +
  geom_raster(alpha=0.7) +
  scale_fill_manual(name = "Fourth ventricle infiltration",
                    values = c('papayawhip','magenta4'),
                    labels = c("No", "Yes","Missing")) +
  theme(legend.position=c(-0.015, 0.5),
        legend.title = element_text(size = 5),
        legend.text = element_text(size = 5),
        legend.key.size = unit(0.5, 'cm'), #change legend key size
        legend.key.height = unit(0.5, 'cm'), #change legend key height
        legend.key.width = unit(0.5, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.5,-0.5,-0.5,1), "cm"))+
  labs(x = "",
       y = "", title = "") +
  #scale_fill_discrete(name = "Molecular subtypes")+
  coord_flip()

row.plot7 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('race', id, fill = df$Race)) +
  geom_raster(alpha=0.9) +
  #legend.position=c(-0.022, 0.5)
  theme(legend.position="bottom",
        legend.title = element_text(size = 5),
        legend.text = element_text(size = 5),
        legend.key.size = unit(0.5, 'cm'), #change legend key size
        legend.key.height = unit(0.5, 'cm'), #change legend key height
        legend.key.width = unit(0.5, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.5,-0.5,-0.5,1), "cm"))+
  labs(x = "",
       y = "", title = "") +
  scale_fill_manual(name = "Race                          ",
                    values = c('mintcream','navyblue','violetred'),
                    labels = c("Asian", "Other","White","Missing")) +
  coord_flip()

tiff("matrixplot.tiff", width=19.75, height=17.75, units="cm", res=300)
grid.draw(rbind(ggplotGrob(row.plot1), ggplotGrob(row.plot2), ggplotGrob(row.plot3),ggplotGrob(row.plot4),ggplotGrob(row.plot5), ggplotGrob(row.plot6), ggplotGrob(row.plot7), size = "min"))
dev.off()
#################################
#matrixplot_1
library(plyr)
library(grid)
df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
df<-subset(df, df[,"mol_sub"]== "SHHTP53-"|df[,"mol_sub"]== "SHHTP53+"|df[,"mol_sub"]== "G3"|df[,"mol_sub"]== "G4"|df[,"mol_sub"]== "WNT")
#df <- df %>% 
 # mutate(HistologicalSubtypes = coalesce(HistologicalSubtypes, 0))
df$mol_sub <- ifelse(df$mol_sub=='WNT','1',df$mol_sub)
df$mol_sub <- ifelse(df$mol_sub=='G3','2',df$mol_sub)
df$mol_sub <- ifelse(df$mol_sub=='G4','3',df$mol_sub)
df$mol_sub <- ifelse(df$mol_sub=='SHHTP53-','4',df$mol_sub)
df$mol_sub <- ifelse(df$mol_sub=='SHHTP53+','5',df$mol_sub)
df$mol_sub <-factor(df$mol_sub)
df$fvi <- ifelse(df$TumorPosition==3,2,df$TumorPosition)
df$Race <- ifelse(df$Race=="More Than One Race"|df$Race=="Native Hawaiian or Other Pacific Islander","Other",df$Race)
#df$Race <- ifelse(is.na(df$Race),"Missing",df$Race)
df$Race <- ifelse(df$Race=='White','cwhite',df$Race)
df$Age <- findInterval(df$Age, c(3.1,18.1))
df$HistologicalSubtypes <-factor(df$HistologicalSubtypes)
df$Sex <-factor(df$Sex)
df$Age <-factor(df$Age)
df$fvi <-factor(df$fvi)
df <-arrange(df, mol_sub,HistologicalSubtypes,Age,Hydrocephalus,TumorTexture,fvi,Sex,Race) 

row.plot1 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('mol', id, fill = df$mol_sub)) +
  geom_tile(alpha=1,colour = 'grey50') +
  scale_fill_manual(name = "Molecular subgroup",
                    values = c('#C5C6C7','#6197D5', '#B92B78','#F19E38','#F1564D'),
                    labels = c('WNT',"G3", "G4",'SHHTP53-','SHHTP53+')) +
  theme(legend.position=c(-0.028, 0.5),
        legend.title = element_text(size = 5,face = "bold"),
        legend.text = element_text(size = 5),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm"))+
  labs(x = "",
       y = "", title = "") +
  #scale_fill_discrete(name = "Molecular subgroup")+
  coord_flip()


row.plot2 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('his', id, fill = df$HistologicalSubtypes)) +
  geom_tile(alpha=1,colour = 'grey50') +
  theme(legend.position=c(-0.025, 0.5),
        legend.title = element_text(size = 5,face = "bold"),
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        #plot.background = element_rect(colour = "grey50", size = 1),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm"))+
  labs(x = "",
       y = "", title = "") +
  scale_fill_manual(name = "Histological subgroup",labels = c('Classic', 'DN', 'MBEN', 'LCA'),values=c("gray", "palegreen4", "steelblue", "firebrick","gray50"),limits = c('1', '2', '3', '4'))+
  #scale_color_manual(values=c(c("lightgreen", "#F0E442", "#0072B2", "red","gray50"))+
  coord_flip()

row.plot3 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('Age', id, fill = df$Age)) +
  geom_tile(alpha=1,colour = 'grey50') +
  scale_fill_manual(name = "Age",
                    values = c('lightcyan1', 'skyblue','blue4','gray50'),
                    labels = c("<=3","3-18", ">18"),limits = c('0', '1','2')) +
  theme(legend.position=c(-0.053, 0.5),
        legend.title = element_text(size = 5,face = "bold"),
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm")
  )+
  labs(x = "",
       y = "", title = "") +
  coord_flip()

row.plot4 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('Hyd', id, fill = df$Hydrocephalus)) +
  geom_tile(alpha=1,colour = 'grey50') +
  scale_fill_manual(name = "Hydrocephalus",
                    values = c( 'gray','#FF7F50'),
                    labels = c("No","Yes")) +
  theme(legend.position=c(-0.041, 0.5),
        legend.title = element_text(size = 5,face = "bold"),
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm")
  )+
  labs(x = "",
       y = "", title = "") +
  coord_flip()

row.plot5 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('texture', id, fill = df$TumorTexture)) +
  geom_tile(alpha=1,colour = 'grey50') +
  scale_fill_manual(name = "Tumor texture",
                    values = c('gray', 'thistle1','purple4','gray50'),
                    labels = c('Soft', "Mix","Hard"),limits = c('Soft','Mix', 'Hard')) +
  theme(legend.position=c(-0.045, 0.5),
        legend.title = element_text(size = 5,face = "bold"),
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm"))+
  labs(x = "",
       y = "", title = "") +
  #scale_fill_discrete(name = "Molecular subtypes")+
  coord_flip()

row.plot6 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('fvi', id, fill = df$fvi)) +
  geom_tile(alpha=1,colour = 'grey50') +
  scale_fill_manual(name = "4th ventricle infiltration",
                    values = c('#12abb0','gray','gray50'),
                    labels = c("No", "Yes"),limits = c('1','2')) +
  theme(legend.position=c(-0.022, 0.5),
        legend.title = element_text(size = 5,face = "bold"),
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        #legend.key = element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm"))+
  labs(x = "",
       y = "", title = "") +
  #scale_fill_discrete(name = "Molecular subtypes")+
  coord_flip()

row.plot7 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('Sex', id, fill = df$Sex)) +
  geom_tile(alpha=1,colour = 'grey50') +
  #legend.position=c(-0.022, 0.5)
  theme(legend.position=c(-0.025, 0.5),
        legend.title = element_text(size = 5,face = "bold"),
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm"))+
  labs(x = "",
       y = "", title = "") +
  scale_fill_manual(name = "Gender                          ",
                    values = c('lightpink','lightblue'),
                    labels = c("Female","Male")) +
  coord_flip()

row.plot8 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('Race', id, fill = df$Race)) +
  geom_tile(alpha=1,colour = 'grey50') +
  #legend.position=c(-0.022, 0.5)
  labs(x = "",
       y = "", title = "")  +
  theme(legend.position=c(0.4, -0.3),legend.direction = "horizontal",
        legend.title = element_blank(),
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_text(face = "bold",angle = 0, vjust = 0, hjust=-2,margin = margin(r = -12),size=5),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm"))+
  
  scale_fill_manual(name = "Race                          ",
                    values = c('mintcream','springgreen4','violetred','#007fff','gray50'),
                    labels = c("Asian","Black or African American","White","Other","Missing")) +
  coord_flip()

#library(ggplot2)
#library(gridExtra)
tiff("deneme.tiff", width=18.75, height=20, units="cm", res=300)
grid.arrange(ggplotGrob(row.plot1), ggplotGrob(row.plot2),ggplotGrob(row.plot3),ggplotGrob(row.plot4),ggplotGrob(row.plot5), ggplotGrob(row.plot6), ggplotGrob(row.plot7),ggplotGrob(row.plot8),nrow=9)
dev.off()

#################################
#option 2
#matrixplot_1
library(plyr)
library(grid)
df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
df<-subset(df, df[,"mol_sub"]== "SHHTP53-"|df[,"mol_sub"]== "SHHTP53+"|df[,"mol_sub"]== "G3"|df[,"mol_sub"]== "G4"|df[,"mol_sub"]== "WNT")
#df <- df %>% 
# mutate(HistologicalSubtypes = coalesce(HistologicalSubtypes, 0))
df$mol_sub <- ifelse(df$mol_sub=='WNT','1',df$mol_sub)
df$mol_sub <- ifelse(df$mol_sub=='G3','2',df$mol_sub)
df$mol_sub <- ifelse(df$mol_sub=='G4','3',df$mol_sub)
df$mol_sub <- ifelse(df$mol_sub=='SHHTP53-','4',df$mol_sub)
df$mol_sub <- ifelse(df$mol_sub=='SHHTP53+','5',df$mol_sub)
df$mol_sub <-factor(df$mol_sub)
df$fvi <- ifelse(df$TumorPosition==3,2,df$TumorPosition)
df$Race <- ifelse(df$Race=="More Than One Race"|df$Race=="Native Hawaiian or Other Pacific Islander","Other",df$Race)
#df$Race <- ifelse(is.na(df$Race),"Missing",df$Race)
df$Race <- ifelse(df$Race=='White','cwhite',df$Race)
df$Age <- findInterval(df$Age, c(3.1,18.1))
df$HistologicalSubtypes <-factor(df$HistologicalSubtypes)
df$Sex <-factor(df$Sex)
df$Age <-factor(df$Age)
df$fvi <-factor(df$fvi)
df <-arrange(df, mol_sub,Age,HistologicalSubtypes,Hydrocephalus,TumorTexture,fvi,Sex,Race) 

row.plot1 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('mol', id, fill = df$mol_sub)) +
  geom_tile(alpha=1,colour = 'grey50') +
  scale_fill_manual(name = "Molecular subgroup",
                    values = c('#C5C6C7','#6197D5', '#B92B78','#F19E38','#F1564D'),
                    labels = c('WNT',"G3", "G4",'SHHTP53-','SHHTP53+')) +
  theme(legend.position=c(-0.028, 0.5),
        legend.title = element_text(size = 5,face = "bold"),
        legend.text = element_text(size = 5),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm"))+
  labs(x = "",
       y = "", title = "") +
  #scale_fill_discrete(name = "Molecular subgroup")+
  coord_flip()


row.plot2 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('his', id, fill = df$HistologicalSubtypes)) +
  geom_tile(alpha=1,colour = 'grey50') +
  theme(legend.position=c(-0.025, 0.5),
        legend.title = element_text(size = 5,face = "bold"),
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        #plot.background = element_rect(colour = "grey50", size = 1),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm"))+
  labs(x = "",
       y = "", title = "") +
  scale_fill_manual(name = "Histological subgroup",labels = c('Classic', 'DN', 'MBEN', 'LCA'),values=c("gray", "palegreen4", "steelblue", "firebrick","gray50"),limits = c('1', '2', '3', '4'))+
  #scale_color_manual(values=c(c("lightgreen", "#F0E442", "#0072B2", "red","gray50"))+
  coord_flip()

row.plot3 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('Age', id, fill = df$Age)) +
  geom_tile(alpha=1,colour = 'grey50') +
  scale_fill_manual(name = "Age",
                    values = c('lightcyan1', 'skyblue','blue4','gray50'),
                    labels = c("<=3","3-18", ">18"),limits = c('0', '1','2')) +
  theme(legend.position=c(-0.053, 0.5),
        legend.title = element_text(size = 5,face = "bold"),
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm")
  )+
  labs(x = "",
       y = "", title = "") +
  coord_flip()

row.plot4 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('Hyd', id, fill = df$Hydrocephalus)) +
  geom_tile(alpha=1,colour = 'grey50') +
  scale_fill_manual(name = "Hydrocephalus",
                    values = c( 'gray','#FF7F50'),
                    labels = c("No","Yes")) +
  theme(legend.position=c(-0.041, 0.5),
        legend.title = element_text(size = 5,face = "bold"),
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm")
  )+
  labs(x = "",
       y = "", title = "") +
  coord_flip()

row.plot5 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('texture', id, fill = df$TumorTexture)) +
  geom_tile(alpha=1,colour = 'grey50') +
  scale_fill_manual(name = "Tumor texture",
                    values = c('gray', 'thistle1','purple4','gray50'),
                    labels = c('Soft', "Mix","Hard"),limits = c('Soft','Mix', 'Hard')) +
  theme(legend.position=c(-0.045, 0.5),
        legend.title = element_text(size = 5,face = "bold"),
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm"))+
  labs(x = "",
       y = "", title = "") +
  #scale_fill_discrete(name = "Molecular subtypes")+
  coord_flip()

row.plot6 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('fvi', id, fill = df$fvi)) +
  geom_tile(alpha=1,colour = 'grey50') +
  scale_fill_manual(name = "4th ventricle infiltration",
                    values = c('#12abb0','gray','gray50'),
                    labels = c("No", "Yes"),limits = c('1','2')) +
  theme(legend.position=c(-0.022, 0.5),
        legend.title = element_text(size = 5,face = "bold"),
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        #legend.key = element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm"))+
  labs(x = "",
       y = "", title = "") +
  #scale_fill_discrete(name = "Molecular subtypes")+
  coord_flip()

row.plot7 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('Sex', id, fill = df$Sex)) +
  geom_tile(alpha=1,colour = 'grey50') +
  #legend.position=c(-0.022, 0.5)
  theme(legend.position=c(-0.025, 0.5),
        legend.title = element_text(size = 5,face = "bold"),
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm"))+
  labs(x = "",
       y = "", title = "") +
  scale_fill_manual(name = "Gender                          ",
                    values = c('lightpink','lightblue'),
                    labels = c("Female","Male")) +
  coord_flip()

row.plot8 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('Race', id, fill = df$Race)) +
  geom_tile(alpha=1,colour = 'grey50') +
  #legend.position=c(-0.022, 0.5)
  labs(x = "",
       y = "", title = "")  +
  theme(legend.position=c(0.4, -0.3),legend.direction = "horizontal",
        legend.title = element_blank(),
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_text(face = "bold",angle = 0, vjust = 0, hjust=-2,margin = margin(r = -12),size=5),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm"))+
  
  scale_fill_manual(name = "Race                          ",
                    values = c('mintcream','springgreen4','violetred','#007fff','gray50'),
                    labels = c("Asian","Black or African American","White","Other","Missing")) +
  coord_flip()

#library(ggplot2)
#library(gridExtra)
tiff("deneme_1.tiff", width=18.75, height=20, units="cm", res=300)
grid.arrange(ggplotGrob(row.plot1), ggplotGrob(row.plot3),ggplotGrob(row.plot2),ggplotGrob(row.plot4),ggplotGrob(row.plot5), ggplotGrob(row.plot6), ggplotGrob(row.plot7),ggplotGrob(row.plot8),nrow=9)
dev.off()

#################################
#option 3
#matrixplot_1
library(plyr)
library(grid)
df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
df<-subset(df, df[,"mol_sub"]== "SHHTP53-"|df[,"mol_sub"]== "SHHTP53+"|df[,"mol_sub"]== "G3"|df[,"mol_sub"]== "G4"|df[,"mol_sub"]== "WNT")
#df <- df %>% 
# mutate(HistologicalSubtypes = coalesce(HistologicalSubtypes, 0))
df$mol_sub <- ifelse(df$mol_sub=='WNT','1',df$mol_sub)
df$mol_sub <- ifelse(df$mol_sub=='SHHTP53-','2',df$mol_sub)
df$mol_sub <- ifelse(df$mol_sub=='SHHTP53+','3',df$mol_sub)
df$mol_sub <- ifelse(df$mol_sub=='G3','4',df$mol_sub)
df$mol_sub <- ifelse(df$mol_sub=='G4','5',df$mol_sub)

df$mol_sub <-factor(df$mol_sub)
df$fvi <- ifelse(df$TumorPosition==3,2,df$TumorPosition)
df$Race <- ifelse(df$Race=="More Than One Race"|df$Race=="Native Hawaiian or Other Pacific Islander","Other",df$Race)
#df$Race <- ifelse(is.na(df$Race),"Missing",df$Race)
df$Race <- ifelse(df$Race=='White','cwhite',df$Race)
df$Age <- findInterval(df$Age, c(3.1,18.1))
df$HistologicalSubtypes <-factor(df$HistologicalSubtypes)
df$Sex <-factor(df$Sex)
df$Age <-factor(df$Age)
df$fvi <-factor(df$fvi)
df <-arrange(df, mol_sub,Age,Hydrocephalus,HistologicalSubtypes,TumorTexture,fvi,Sex,Race) 

row.plot1 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('mol', id, fill = df$mol_sub)) +
  geom_tile(alpha=1,colour = 'grey50') +
  scale_fill_manual(name = "Molecular subgroup",
                    values = c('#C5C6C7','#F19E38','#F1564D','#6197D5', '#B92B78'),
                    labels = c('WNT','SHHTP53-','SHHTP53+',"G3", "G4")) +
  theme(legend.position=c(-0.028, 0.5),
        legend.title = element_text(size = 5,face = "bold"),
        legend.text = element_text(size = 5),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm"))+
  labs(x = "",
       y = "", title = "") +
  #scale_fill_discrete(name = "Molecular subgroup")+
  coord_flip()


row.plot2 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('his', id, fill = df$HistologicalSubtypes)) +
  geom_tile(alpha=1,colour = 'grey50') +
  theme(legend.position=c(-0.025, 0.5),
        legend.title = element_text(size = 5,face = "bold"),
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        #plot.background = element_rect(colour = "grey50", size = 1),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm"))+
  labs(x = "",
       y = "", title = "") +
  scale_fill_manual(name = "Histological subgroup",labels = c('Classic', 'DN', 'MBEN', 'LCA'),values=c("gray", "palegreen4", "steelblue", "firebrick","gray50"),limits = c('1', '2', '3', '4'))+
  #scale_color_manual(values=c(c("lightgreen", "#F0E442", "#0072B2", "red","gray50"))+
  coord_flip()

row.plot3 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('Age', id, fill = df$Age)) +
  geom_tile(alpha=1,colour = 'grey50') +
  scale_fill_manual(name = "Age",
                    values = c('lightcyan1', 'skyblue','blue4','gray50'),
                    labels = c("<=3","3-18", ">18"),limits = c('0', '1','2')) +
  theme(legend.position=c(-0.053, 0.5),
        legend.title = element_text(size = 5,face = "bold"),
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm")
  )+
  labs(x = "",
       y = "", title = "") +
  coord_flip()

row.plot4 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('Hyd', id, fill = df$Hydrocephalus)) +
  geom_tile(alpha=1,colour = 'grey50') +
  scale_fill_manual(name = "Hydrocephalus",
                    values = c( 'gray','#FF7F50'),
                    labels = c("No","Yes")) +
  theme(legend.position=c(-0.041, 0.5),
        legend.title = element_text(size = 5,face = "bold"),
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm")
  )+
  labs(x = "",
       y = "", title = "") +
  coord_flip()

row.plot5 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('texture', id, fill = df$TumorTexture)) +
  geom_tile(alpha=1,colour = 'grey50') +
  scale_fill_manual(name = "Tumor texture",
                    values = c('gray', 'thistle1','purple4','gray50'),
                    labels = c('Soft', "Mix","Hard"),limits = c('Soft','Mix', 'Hard')) +
  theme(legend.position=c(-0.045, 0.5),
        legend.title = element_text(size = 5,face = "bold"),
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm"))+
  labs(x = "",
       y = "", title = "") +
  #scale_fill_discrete(name = "Molecular subtypes")+
  coord_flip()

row.plot6 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('fvi', id, fill = df$fvi)) +
  geom_tile(alpha=1,colour = 'grey50') +
  scale_fill_manual(name = "4th ventricle infiltration",
                    values = c('#12abb0','gray','gray50'),
                    labels = c("No", "Yes"),limits = c('1','2')) +
  theme(legend.position=c(-0.022, 0.5),
        legend.title = element_text(size = 5,face = "bold"),
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        #legend.key = element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm"))+
  labs(x = "",
       y = "", title = "") +
  #scale_fill_discrete(name = "Molecular subtypes")+
  coord_flip()

row.plot7 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('Sex', id, fill = df$Sex)) +
  geom_tile(alpha=1,colour = 'grey50') +
  #legend.position=c(-0.022, 0.5)
  theme(legend.position=c(-0.025, 0.5),
        legend.title = element_text(size = 5,face = "bold"),
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm"))+
  labs(x = "",
       y = "", title = "") +
  scale_fill_manual(name = "Gender                          ",
                    values = c('lightpink','lightblue'),
                    labels = c("Female","Male")) +
  coord_flip()

row.plot8 <- df %>%
  dplyr::mutate(id = row_number()) %>%
  ggplot(aes('Race', id, fill = df$Race)) +
  geom_tile(alpha=1,colour = 'grey50') +
  #legend.position=c(-0.022, 0.5)
  labs(x = "",
       y = "", title = "")  +
  theme(legend.position=c(0.4, -0.3),legend.direction = "horizontal",
        legend.title = element_blank(),
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, 'cm'), #change legend key size
        legend.key.height = unit(0.3, 'cm'), #change legend key height
        legend.key.width = unit(0.3, 'cm'),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_text(face = "bold",angle = 0, vjust = 0, hjust=-2,margin = margin(r = -12),size=5),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.margin=unit(c(-0.1,-0.5,-0.5,1), "cm"))+
  
  scale_fill_manual(name = "Race                          ",
                    values = c('mintcream','springgreen4','violetred','#007fff','gray50'),
                    labels = c("Asian","Black or African American","White","Other","Missing")) +
  coord_flip()

#library(ggplot2)
#library(gridExtra)
tiff("deneme_2.tiff", width=18.75, height=20, units="cm", res=300)
grid.arrange(ggplotGrob(row.plot1), ggplotGrob(row.plot3),ggplotGrob(row.plot4),ggplotGrob(row.plot2),ggplotGrob(row.plot5), ggplotGrob(row.plot6), ggplotGrob(row.plot7),ggplotGrob(row.plot8),nrow=9)
dev.off()

#################
#results 
df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
surv.object<-Surv(df$OS_nomissing_adjusted, df$mortality)
km.by<-npsurv(surv.object ~ 1,data = df, conf.type = "log-log")
print(km.by)
#5 years survival rates
summary(km.by, times=seq(0,60,60))

df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
df<-subset(df, df[,"mol_sub"]== "G4")
surv.object<-Surv(df$OS_nomissing_adjusted, df$mortality)
km.by<-npsurv(surv.object ~ 1,data = df, conf.type = "log-log")
print(km.by)
#5 years survival rates
summary(km.by, times=seq(0,60,60))

library(tableone)
df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
df<-subset(df, df[,"mol_sub"]== "SHHTP53+"|df[,"mol_sub"]== "SHHTP53-")
factorVars1 <- c("Sex","HistologicalSubtypes","HistologicalSubtypes_2","TumorTexture","Hydrocephalus","TumorPosition_2")
vars1 <- c("Age","Sex","HistologicalSubtypes","HistologicalSubtypes_2","TumorTexture","Hydrocephalus","TumorPosition_2")
table1 <- CreateTableOne(vars = vars1, strata = c("mol_sub"), data = df, factorVars = factorVars1)
table1 <- print(table1,smd = TRUE)

df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )
df<-subset(df, df[,"mol_sub"]== "G3"|df[,"mol_sub"]== "G4")
factorVars1 <- c("Sex","HistologicalSubtypes","HistologicalSubtypes_2","TumorTexture","Hydrocephalus","TumorPosition_2")
vars1 <- c("Age","Sex","HistologicalSubtypes","HistologicalSubtypes_2","TumorTexture","Hydrocephalus","TumorPosition_2")
table1 <- CreateTableOne(vars = vars1, strata = c("mol_sub"), data = df, factorVars = factorVars1)
table1 <- print(table1,smd = TRUE)
