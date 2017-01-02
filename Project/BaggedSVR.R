rm(list=ls())

library(kernlab)
library(doMC)
library(plyr)
registerDoMC(4)

set.seed(1337)
setwd('/Users/krishna/MIRI/KML')
source('github/Project/Utils.R')
train = read.csv('data/train.csv')
test = read.csv('data/test.csv')
data = fix_levels(train,test)
train = data[[1]]
test = data[[2]]
train$loss = log(train$loss)
names(train$loss) = c('loss')

pca_train =  read.csv('data/pca/pca_train_comp.csv')
datatrain_stack = data.frame(pca_train,train)

train_feat = data.frame(model.matrix(~ . - 1, datatrain_stack))

importance = read.csv('/Users/krishna/MIRI/KML/data/featuers/varimp.csv')
top_50 = importance$variable[1:50]
top_50_train = train_feat[,c(top_50)]
loss = train_feat[,c('loss')]
train_data = data.frame(top_50_train,loss)

# Predict
pca_test =  read.csv('data/pca/pca_test_comp.csv')
datatest_stack = data.frame(pca_test,test)
test_feat = data.frame(model.matrix(~ . - 1, datatest_stack))
top_50_tests = test_feat[,c(top_50)]


formual1 = as.formula('loss ~ .')
OBS = 5000
BAG = 30
y_hats = matrix(nrow = nrow(test),ncol = BAG)
for(i in 1:BAG){
  train_sample = train_data[sample(nrow(train_data), OBS), ]
  svr_model = kfold_svr(train_sample, formuale=formual1)
  y_hat = predict(svr_model[[1]],top_50_tests)
  y_hats[,i] = exp(y_hat)
}

avg_yhat = rowMeans(y_hats)
output = data.frame(test$id, round(avg_yhat,4))
names(output) = c('id','loss')

write.table(output,"data/models/svr_bag_30_5000obs.csv", row.names=FALSE,
            sep=",",quote=F)
