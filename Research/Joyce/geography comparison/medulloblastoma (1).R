#######################################
# Author: Fatma Gunturkun
# Date started: 12/2/23
# Last updated: 12/2/23
######################################
################################################################################
#missing_data_imputation
# load packages 
library(mice)

#read dataset
df_1 <- read.csv("./data.csv", stringsAsFactors=FALSE, na.strings = "" )
impute.index  <- df_1$index_os
impute.index  <- impute.index[!is.na(impute.index)]
df_1$Date_of_Surgery <- as.Date(df_1$Date_of_Surgery,format = "%d-%b-%y")
df_1$time <- difftime(df_1$Date_of_Surgery, min(df_1$Date_of_Surgery, na.rm = TRUE), units = "days") / 30
df_1$time <- as.numeric(df_1$time)

df_1 <- df_1[,c("Geographical_Location","Race","Age","Sex","time","Molecular_Subtypes","Histological_Subtypes","Hydrocephalus_before_Surgery","Tumor_Texture","Fourth_ventricle_infiltration","Survival_status","OS_missing","DFS_missing","Recurrence_label","Radiotherapy","Chemotherapy","TumorResection","Leptomeningeal_Dissemination")]
train.impute <- df_1
train.impute$Geographical_Location<-factor(train.impute$Geographical_Location)
train.impute$Race<-factor(train.impute$Race)
train.impute$Sex<-factor(train.impute$Sex)
train.impute$Molecular_Subtypes<-factor(train.impute$Molecular_Subtypes)
train.impute$Histological_Subtypes<-factor(train.impute$Histological_Subtypes)
train.impute$Hydrocephalus_before_Surgery<-factor(train.impute$Hydrocephalus_before_Surgery)
train.impute$Tumor_Texture<-factor(train.impute$Tumor_Texture)
train.impute$Fourth_ventricle_infiltration<-factor(train.impute$Fourth_ventricle_infiltration)
train.impute$Survival_status<-factor(train.impute$Survival_status)
train.impute$Recurrence_label<-factor(train.impute$Recurrence_label)
train.impute$Radiotherapy<-factor(train.impute$Radiotherapy)
train.impute$Chemotherapy<-factor(train.impute$Chemotherapy)
train.impute$TumorResection<-factor(train.impute$TumorResection)
train.impute$Leptomeningeal_Dissemination<-factor(train.impute$Leptomeningeal_Dissemination)

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
train.imputed$Geographical_Location<-factor(train.imputed$Geographical_Location)
train.imputed$Race<-factor(train.imputed$Race)
train.imputed$Sex<-factor(train.imputed$Sex)
train.imputed$Molecular_Subtypes<-factor(train.imputed$Molecular_Subtypes)
train.imputed$Histological_Subtypes<-factor(train.imputed$Histological_Subtypes)
train.imputed$Hydrocephalus_before_Surgery<-factor(train.imputed$Hydrocephalus_before_Surgery)
train.imputed$Tumor_Texture<-factor(train.imputed$Tumor_Texture)
train.imputed$Fourth_ventricle_infiltration<-factor(train.imputed$Fourth_ventricle_infiltration)
train.imputed$Survival_status<-factor(train.imputed$Survival_status)
train.imputed$Recurrence_label<-factor(train.imputed$Recurrence_label)
train.imputed$Radiotherapy<-factor(train.imputed$Radiotherapy)
train.imputed$Chemotherapy<-factor(train.imputed$Chemotherapy)
train.imputed$TumorResection<-factor(train.imputed$TumorResection)
train.imputed$Leptomeningeal_Dissemination<-factor(train.imputed$Leptomeningeal_Dissemination)


miceimp <- mice(data = train.imputed, m = 1,maxit = 10, method = "pmm", printFlag = F)
complete1 <- complete(miceimp,1)
write.csv(complete1,'complete1.csv')


################################################################################
#comparison of the cohorts
df <- read.csv("./data.csv", stringsAsFactors=FALSE, na.strings = "" )
factorVars1 <- c("Race","Sex","Molecular_Subtypes","Histological_Subtypes","Genetic_Mutations","Hydrocephalus_before_Surgery","Tumor_Texture","Fourth_ventricle_infiltration","Survival_status","Recurrence_label","Radiotherapy","Chemotherapy","TumorResection","Leptomeningeal_Dissemination")
vars1 <- c("Race","Age","Sex","Molecular_Subtypes","Histological_Subtypes","Genetic_Mutations","Hydrocephalus_before_Surgery","Tumor_Texture","Fourth_ventricle_infiltration","Survival_status","OS_nomissing_adjusted","Recurrence_label","Radiotherapy","Chemotherapy","TumorResection","Leptomeningeal_Dissemination")
# Convert variables to factors and add a level for missing values
for (var in factorVars1) {
  df[[var]] <- factor(df[[var]], levels = unique(df[[var]]))
}

# Add a level for missing values
#for (var in factorVars1) {
#  df[[var]] <- factor(df[[var]], levels = c(levels(df[[var]]), "Missing"))
#  df[[var]][is.na(df[[var]])] <- "Missing"
#}

table1 <- CreateTableOne(vars = vars1, strata = c("Geographical_Location"), data = df, factorVars = factorVars1)
table1 <- print(table1,smd = TRUE,showAllLevels = TRUE)
write.csv(table1, "./table1_china_us1.csv")
t_test_result <- t.test(Age ~ Geographical_Location, data = df)
print(t_test_result)

##########################
df <- read.csv("./data.csv", stringsAsFactors=FALSE, na.strings = "" )
factorVars1 <- c("Race","Sex","Molecular_Subtypes","Histological_Subtypes","Genetic_Mutations","Hydrocephalus_before_Surgery","Tumor_Texture","Fourth_ventricle_infiltration","Survival_status","Recurrence_label","Radiotherapy","Chemotherapy","TumorResection","Leptomeningeal_Dissemination")
vars1 <- c("Race","Age","Sex","Molecular_Subtypes","Histological_Subtypes","Genetic_Mutations","Hydrocephalus_before_Surgery","Tumor_Texture","Fourth_ventricle_infiltration","Survival_status","OS_nomissing_adjusted","Recurrence_label","Radiotherapy","Chemotherapy","TumorResection","Leptomeningeal_Dissemination")
# Convert variables to factors and add a level for missing values
for (var in factorVars1) {
  df[[var]] <- factor(df[[var]], levels = unique(df[[var]]))
}
# Add a level for missing values
#for (var in vars1) {
#  df[[var]] <- factor(df[[var]], levels = c(levels(df[[var]]), "Missing"))
#  df[[var]][is.na(df[[var]])] <- "Missing"
#}
table1 <- CreateTableOne(vars = vars1, strata = c("internal_external"), data = df, factorVars = factorVars1)
table1 <- print(table1,smd = TRUE,showAllLevels = TRUE)
write.csv(table1, "./table1_internal_external1.csv")
t_test_result <- t.test(Age ~ internal_external, data = df)
print(t_test_result)

