library(data.table)
library(caTools)
library(doMC)
library(caret)
library(glmnet)
registerDoMC(4)
set.seed(1337)
setwd('/Users/krishna/MIRI/KML')

TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"
SUBMISSION_FILE = "data/output.csv"
train = fread(TRAIN_FILE, showProgress = TRUE)
train_df = data.frame(train)
train_df$loss = log(train_df$loss)

# All Numerical Values
numeric_col = sapply(train_df, is.numeric) 
datatrain_num = train_df[,numeric_col]

remove_col=c('id')
tr = datatrain_num[, -which(names(datatrain_num) %in% remove_col)]

sqtr = data.frame(apply(tr[1:14],2,function(x)x^2))
names(sqtr) <- paste0(names(sqtr), "_sq")
cubetr = data.frame(apply(tr[1:14],2,function(x)x^3))
names(cubetr) <- paste0(names(cubetr), "_cube")
logtr = data.frame(apply(tr[1:14],2,function(x)log(x)))
names(logtr) <- paste0(names(logtr), "_log")
exptr = data.frame(apply(tr[1:14],2,function(x)exp(x)))
names(exptr) <- paste0(names(exptr), "_exp")
pca_train = read.csv('data/pca/pca_train_comp.csv')

tr_all = data.frame(sqtr,cubetr,logtr,exptr,pca_train,tr)

library(h2oEnsemble)
conn = h2o.init()
train = as.h2o(x=tr_all,destination_frame = "train.hex")
str(tr_all)
feat = names(tr_all)[1:ncol(tr_all)- 1]
label= names(tr_all)[ncol(tr_all)]
rf1 = h2o.randomForest(x=feat, y=label, training_frame = train)
m = data.frame(h2o.varimp(rf1))
plot(1:70,m$percentage, pch=19, cex=0.9, col='blue')
#text(1:70,m$percentage+0.001,labels=m$variable, cex=0.9)
# Choose Best Featuers based on RF Importance
# Choose Best Features based on Elastic Net

model1 = glmnet(tr,tr$loss)
?glmnet
