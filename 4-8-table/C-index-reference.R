library(survC1)
df <- read.csv("your-data-file.csv", stringsAsFactors=FALSE, na.strings = "" )
data <- data.frame(df$ClassToCompare, df$censored, df$OS)
colnames(data) <- c("subtype", "censored", "OS")
N = 1000
C_index = rep(0, N)
n = nrow(data)
for (i in 1:N){
  if(i %% 100 == 0) cat(i, "times of trying...\n")
  # redevide the train and test data in each epoch
  set.seed(i)
  index.training=sample(n, 0.8 * n, F)
  index.testing=(1:n)[-index.training]
  
  res.cox <- coxph(Surv(OS, censored) ~ relevel(factor(subtype), ref = "group-referred-to"), data = data[index.training, ])
  # test on the test_data
  marker <- as.vector(predict(res.cox, data[index.testing, ], type = 'lp'))
  
  tau=quantile(data$OS[index.testing], 0.95)
  D=cbind(data$OS[index.testing], data$censored[index.testing], marker)
  C_index[i] = Inf.Cval(D, tau, itr=200)$Dhat
}
print(C_index[1:10])
cat("The mean of C index in", N, "epochs =", mean(C_index))