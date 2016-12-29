rm(list=ls())

library(plsdepot)
library(CatEncoders)
require(caTools)

setwd('/Users/krishna/MIRI/MVA/Assignment2')

train = read.csv('../zip_data/zip_train.dat',header = F, sep=" ")
test = read.csv('../zip_data/zip_test.dat',header = F, sep=" ")

head(train)

# As V258 has NAs
remove_col = c('V258')
train = train[,!names(train) %in% remove_col]
test = test[,!names(test) %in% remove_col]
ytrain = data.frame(train$V1)
ytest = data.frame(test$V1)

split = sample.split(train$V1, 0.05)
table(split)
# Stack data and center
remove_col = c('V1')
data = rbind(train,test)
data = scale(data, center = TRUE, scale = FALSE)

# Split Data
trainIB = data[1:nrow(train),]
testIB = data[-c(1:nrow(train)),]
split = sample.split(train$V1, 0.05)


# OHE
ohe_model = OneHotEncoder.fit(data.frame(ytrain))
ytrain_ohe = as.matrix(transform(ohe_model,ytrain))
ytrain_ohe = as.data.frame(ytrain_ohe)
names(ytrain_ohe) = c("C0","C1","C2","C3","C4","C5","C6","C7","C8","C9")

# IBA Manual Train
source('Util.R')

trainIBs = subset(trainIB, split ==T)
trainy = subset(ytrain_ohe, split ==T)

TrainIB = interbatt(trainIBs,trainy)[[1]][,1:9]
AdjRsq =  matrix(9)
R2CV = matrix(9)

# When data is centered we need to remove the intercept
formula1 = cbind(C0,C1,C2,C3,C4,C5,C6,C7,C8,C9) ~  0 + . 

n = nrow(TrainIB)
for(i in (1:9)){
  train_data = data.frame(TrainIB[,1:i], trainy)
  model1 = lm(formula1, data=train_data)
  adjr = sapply(summary(model1), function(x){x$adj.r.squared})
  AdjRsq[i] = mean(adjr)
  PRESS  = apply((as.data.frame(model1$residuals)/(1-ls.diag(model1)$hat))^2,2,sum)
  RMPRESS = sqrt(PRESS/n)
  R2cv_denom  = apply(trainy,2,var)*(n-1)
  R2cv = 1 - PRESS/R2cv_denom
  rcv =mean(R2cv) 
  R2CV[i] = rcv
}

plot(1:9,AdjRsq,pch=19,col="red",cex=.7,xlab="Components",ylab="Adj RSquared")
plot(1:9,R2CV,pch=19,col="blue",cex=.7,xlab="Components",ylab="R^2 CV")

# We choose 9 components based on training R square error 0.54
train_data = data.frame(TrainIB[,1:9], trainy)
model1 = lm(formula1, data=train_data)

# IBA Test
A = interbatt(trainIBs,trainy)[[2]][,1:9]
TestIB = (as.matrix(testIB) %*% A)
yhat = predict(model1,data.frame(TestIB))
Yhat = data.frame(unname(apply(yhat, 1, which.max)) - 1)
eval_func(unlist(ytest),unlist(Yhat),cm_show = T)

# IBA on Full Data
TrainIB = interbatt(train,ytrain_ohe)[[1]][,1:9]
train_data = data.frame(TrainIB, ytrain_ohe)
model2 = lm(formula1, data=train_data)
adjr = sapply(summary(model1), function(x){x$adj.r.squared})

A = interbatt(train,ytrain_ohe)[[2]][,1:9]
TestIB = (as.matrix(testIB) %*% A)
yhat = predict(model2,data.frame(TestIB))
Yhat = data.frame(unname(apply(yhat, 1, which.max)) - 1)
eval_func(unlist(ytest),unlist(Yhat),cm_show = T)


# Prepare data for NN
X =  interbatt(trainIBs,trainy)[[1]][,1:9]
yindex = rownames(trainy)
y = ytrain[yindex,]
exportNN = data.frame(X,y)
head(exportNN)
write.table(exportNN,"opNN.csv", row.names=FALSE,sep=",",quote=F)
