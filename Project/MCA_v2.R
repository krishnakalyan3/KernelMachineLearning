rm(list=ls())

source('Utils.R')
setwd('/Users/krishna/MIRI/KML/github/Project - Kaggle')
library(MASS)
set.seed(1337)

train = read.csv('data/train.csv')
test = read.csv('data/test.csv')

data = fix_levels(train,test)
train = data[[1]]
test = data[[2]]
train$loss = log(train$loss)

remove_col=c('id','loss')
datatrain = train[,-which(names(train) %in% remove_col)]
datatest =  test[,-which(names(test) %in% remove_col)]
cat_col = sapply(datatrain, is.factor)
cat_train = datatrain[,cat_col]
cat_test = datatest[,cat_col]

cats = apply(cat_train, 2, function(x) nlevels(as.factor(x)))
cats_17 = cats > 17

mca_train = cat_train[,cats_17]
mca_test = cat_test[,cats_17]

mca = mca(mca_train)
mca_train_comp = data.frame(predict(mca,mca_train))
mca_test_comp = data.frame(predict(mca,mca_test))

saveRDS(mca, 'data/mca/res_res.rds')
write.csv(mca_train_comp, "data/mca/mca_train_comp.csv", row.names=FALSE)
write.csv(mca_test_comp, "data/mca/mca_test_comp.csv", row.names=FALSE)