library(h2oEnsemble)
library(h2o)
library(doMC)
library(flashClust)
registerDoMC(4)
set.seed(1337)
setwd('/Users/krishna/MIRI/KML')

source('/github/Project/Utils.R')
train = read.csv('data/train.csv')
test = read.csv('data/test.csv')

data = fix_levels(train,test)
train = data[[1]]
test = data[[2]]
train$loss = log(train$loss)
names(train$loss) = c('loss')

remove_col=c('id','loss')
tr = train[, -which(names(train) %in% remove_col)]
tr_dummy = model.matrix(~ ., train)

conn = h2o.init()
train_df = as.h2o(x=tr_dummy,destination_frame = "train.hex")
model_kmeans = h2o.h(train_df,k = 100, estimate_k=T)

clust = predict(model_kmeans, test)

output = data.frame(cbind(paste(test$id), clust))
names(output) = c('id','clust')
write.table(output,"clust.csv", row.names=FALSE,
            sep=",",quote=F)