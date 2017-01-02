library(data.table)
library(caTools)
library(doMC)
library(caret)
library(glmnet)
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

remove_col=c('id')
train_feat = datatrain_stack[, -which(names(datatrain_stack) %in% remove_col)]
train_feat = data.frame(model.matrix(~ . - 1, train_feat))

importance = read.csv('/Users/krishna/MIRI/KML/data/featuers/varimp.csv')
top_50 = importance$variable[1:50]
top_50_train = train_feat[,c(top_50)]
top_50_target = train_feat[,c('loss')]
train_data = data.frame(top_50_train,top_50_target)
train = as.h2o(x=train_data,destination_frame = "train.hex")

library(h2oEnsemble)
feat = names(train)[1:ncol(train)- 1]
label= names(train)[ncol(train)]
conn = h2o.init()
rf1 = h2o.randomForest(x=feat, y=label, training_frame = train)
saveRDS(rf1, 'data/models/rf_h2o.rds')

# Test 
pca_test =  read.csv('data/pca/pca_test_comp.csv')
datatest_stack = data.frame(pca_test,test)
test_feat = data.frame(model.matrix(~ . - 1, datatest_stack))
top_50_tests = test_feat[,c(top_50)]
test_df = as.h2o(x=top_50_tests,destination_frame = "test.hex")

yhat = predict(rf1, test_df)
yhat = as.data.frame(exp(yhat))

output = data.frame(test$id, round(yhat,4))
names(output) = c('id','loss')

write.table(output,"data/models/rf.csv", row.names=FALSE,
            sep=",",quote=F)