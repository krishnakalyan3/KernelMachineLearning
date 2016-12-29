library(kernlab)
library(caTools)

setwd('/Users/krishna/MIRI/KML/github/Project - Kaggle')
train = read.csv('data/train.csv')
test = read.csv('data/test.csv')
y = log(train$loss)


remove_col=c('id','loss')
train_data = train[,-which(names(train) %in% remove_col)]


split = sample.split(y, SplitRatio = 0.10)
table(split)


model1 = rvm(train[split,], train$y[split,])
