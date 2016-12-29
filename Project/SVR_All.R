rm(list=ls())

library(kernlab)
library(doMC)
registerDoMC(4)
setwd('/Users/krishna/MIRI/KML/github/Project - Kaggle')

source('Utils.R')
source('CV.R')
set.seed(1337)

train = read.csv('data/train.csv')
test = read.csv('data/test.csv')

data = fix_levels(train,test)
train = data[[1]]
test = data[[2]]
train$loss = log(train$loss)
names(train$loss) = c('loss')

obs = 500
shuffle = TRUE
formual1 = as.formula('loss ~ . -id')

obs = c(500, 1000, 2000, 4000, 8000, 16000)
time_all = c()
mae = c()
for(i in obs){
  ptm = proc.time()
  train_sample = sample(train[1:i,])
  svr_model = kfold_svr(train_sample, formuale=formual1)
  tn = proc.time() - ptm
  time_all= c(time_all, tn[3])
  mae = c(mae, svr_model[[2]])
}

par(mfrow=c(1,2))
plot(obs,mae, pch=19, col="blue", xlab="Observation", ylab="Mean Abasolute Error (MAE)",
     cex=0.9, type='p')
plot(obs,time_all, pch=19, col='red', xlab="Observation", ylab="Time Elapsed (Seconds)",
     cex=0.9,type='p')

svr_model = kfold_svr(train[1:500,], formuale=formual1)
yhat = predict(svr_model, test)
yhat = exp(yhat)

output = data.frame(cbind(paste(test$id), round(yhat,4)))
names(output) = c('id','loss')
write.table(output,"output_svr_1k.csv", row.names=FALSE,
            sep=",",quote=F)