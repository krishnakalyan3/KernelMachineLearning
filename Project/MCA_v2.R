rm(list=ls())

setwd('/Users/krishna/MIRI/KML')

source('github/Project/Utils.R')
library(FactoMineR)
library(doMC)
registerDoMC(4)
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

mca_train = cat_train
mca_test = cat_test

mca = MCA(mca_train, ncp=50)
plot(mca,invisible=c("ind","var"),hab="quali")

mca_train_comp = data.frame(predict(mca, mca_train))
mca_test_comp = data.frame(predict(mca,mca_test))

saveRDS(mca, 'data/mca/res_res.rds')
write.csv(mca_train_comp, "data/mca/mca_train_comp.csv", row.names=FALSE)
write.csv(mca_test_comp, "data/mca/mca_test_comp.csv", row.names=FALSE)