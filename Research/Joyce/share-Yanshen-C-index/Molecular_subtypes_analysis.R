#######################################
# Author: QSU
# Date started: 04/13/23
# Last updated: 04/13/23
######################################

# load packages 
library(survival)
library(KMsurv)
library(dplyr)
library(rms)
library(multcomp)
library(mice)

#############################################
#survival for molecular subtypes

df <- read.csv("./Medulloblastoma-molecular-subtypes.csv", stringsAsFactors=FALSE, na.strings = "" )

#relabel age
df$Age_nomissing <- findInterval(df$Age_nomissing, c(6,18))
df$OS_nomissing_adjusted <- as.numeric(df$OS_nomissing_adjusted)
df$mol_sub <- as.factor(df$mol_sub)

#coxph model
surv.object<-Surv(df$OS_nomissing_adjusted, df$mortality)

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

### specify all pair-wise comparisons among levels of "mol_sub"
sm.glht <- glht(sm, mcp(mol_sub = "Tukey"))
summary(sm.glht)

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


###approach 3, calculate C-index

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
C_index1 = rep(0, N)
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
  C_index1[i] = 1-Inf.Cval(D, tau, itr=200)$Dhat
}
print(C_index1[1:10])
cat("The mean of survC1 C index in", N, "epochs =", mean(C_index1))



Harrell C index

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
  C_index2[i] = index_i
}

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
```

survC1 C index

N = 1000
C_index3 = rep(0, N)
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
  C_index3[i] = 1-Inf.Cval(D, tau, itr=200)$Dhat
}
print(C_index3[1:10])
cat("The mean of survC1 C index in", N, "epochs =", mean(C_index3))


#Harrell C index

C_index4 = rep(0, N)
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
  C_index4[i] = max(index_i, 1-index_i)
}
print(C_index4[1:10])
cat("The mean of Harrell C index in", N, "epochs =", mean(C_index4))

quantile(C_index1, probs = c(.025,0.5,.975))
quantile(C_index2, probs = c(.025,0.5,.975))
quantile(C_index3, probs = c(.025,0.5,.975))
quantile(C_index4, probs = c(.025,0.5,.975))
