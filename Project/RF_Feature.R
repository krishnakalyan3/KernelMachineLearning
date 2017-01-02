library(data.table)
library(caTools)
library(doMC)
library(caret)
library(glmnet)
registerDoMC(4)
set.seed(1337)
setwd('/Users/krishna/MIRI/KML')

# Load all data
TRAIN_FILE = "data/train.csv"
PCA_TRAIN_FILE = "/Users/krishna/MIRI/KML/data/pca/pca_train_comp.csv"
train = data.frame(fread(TRAIN_FILE, showProgress = TRUE))
train$loss = log(train$loss)
pca_train =  data.frame(fread(PCA_TRAIN_FILE, showProgress = TRUE))

datatrain_stack = data.frame(pca_train,train)

remove_col=c('id')
train_feat = datatrain_stack[, -which(names(datatrain_stack) %in% remove_col)]
train_feat = model.matrix(~ ., train_feat)


library(h2oEnsemble)
conn = h2o.init()
train = as.h2o(x=train_feat,destination_frame = "train.hex")
feat = names(train)[1:ncol(train)- 1]
label= names(train)[ncol(train)]

rf1 = h2o.randomForest(x=feat, y=label, training_frame = train)
importance = data.frame(h2o.varimp(rf1))

# Save
saveRDS(rf1, 'data/featuers/rf1.rds')
write.csv(importance, "data/featuers/varimp.csv", row.names=FALSE)

plot(importance$percentage, pch=19,col='blue', cex=0.7, xlab='Number of Featuers'
     ,ylab='Importance')
abline(v=50,col='red')

sum(importance$percentage[1:50])
