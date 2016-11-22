rm(list=ls())
require(caTools)
library(caret)
library(CatEncoders)
require(miscTools)
library(pls)
setwd('/Users/krishna/MIRI/MVA/Assignment1')
source('utils.R')
set.seed(123)

# 1 Read Data with 5 percent
train = read.delim('../ZIP data/zip_train.dat', header = F, sep=" ") 
test = read.delim('../ZIP data/zip_test.dat', header = F, sep=" ") 
remove_col = c('V258')
train = train[,!names(train) %in% remove_col]
test = test[,!names(test) %in% remove_col]

split = sample.split(train$V1, 0.05)
train_split = subset(train, split ==T)

# 2 Define X, Y and Scale
X = data.frame(train_split[,-1])
X = scale(X, center = TRUE, scale = FALSE)
y = data.frame(train_split[,1])
ohe_model = OneHotEncoder.fit(y)
y_ohe = as.matrix(transform(ohe_model,y))
y_ohe = as.data.frame(y_ohe)
names(y_ohe) = c("C0","C1","C2","C3","C4","C5","C6","C7","C8","C9")

# 3 Perform Multi, Compute Avg R sq
train_data = data.frame(X,y_ohe)
formula1 = cbind(C0,C1,C2,C3,C4,C5,C6,C7,C8,C9) ~ . -1
model1 = lm(formula1, data = train_data)
print(paste("R Squared Train",(summary(model1)[[1]])$r.squared))

# 4 Compute LOOCV
PRESS  <- colSums((model1$residuals/(1-ls.diag(model1)$hat))^2)
R2cv   <- 1 - PRESS/(diag(var(y_ohe))*(nrow(X)-1))
print(paste("Avg LOOCV R^2 Train",mean(R2cv)))

# 5 Predict with Centering the average R2 in the test data
Xp = data.frame(test[,-1])
Yp = data.frame(test[,1])
Xp = scale(Xp, center = TRUE, scale = FALSE)
test_scale = data.frame(Xp,Yp)
Yhat = predict(model1,test_scale)
RSS = colSums((Yp-Yhat)^2)
TSS = apply(Yp,2,function(x){sum((x-mean(x))^2)})
r2 = mean(1 - (RSS/TSS))
r2

# 6 Assign every test individual to the maximum response 
Yp_hat = data.frame(unname(apply(Yhat, 1, which.max)) - 1)
eval = eval_func(unlist(Yp),unlist(Yp_hat), cm_show = T)
print(paste("Accuracy :",eval[1]))
print(paste("Error :",eval[2]))

# 7 Perform a PCR without LOO
ncomp = 255
model2 = pcr(formula1, data = train_data, ncomp = ncomp)
Yhat = predict(model2,test)
Y = data.frame(test[,1])
pred_dim = dim(Yhat)
accuracy_metrics = c()
error_metric = c()
for(i in 1:pred_dim[3]){
  Yhat_class = data.frame(unname(apply(Yhat[,,i], 1, which.max)) - 1)
  eval = eval_func(unlist(Y),unlist(Yhat_class))
  accuracy_metrics = c(accuracy_metrics, eval[1])
  error_metric = c(error_metric, eval[2])
}

line_fix =  which.min(error_metric)
print(paste("Component ",line_fix))
print(paste("Error ",min(error_metric)))
qplot(1:pred_dim[3],error_metric, 
      xlab="Number of Components", ylab="Error",
      main="PCR Error Plot", geom='line') +
  geom_vline(xintercept = line_fix,  color = "red", linetype="dotted")

# 8 PCR with LOOCV with 60 Components 
model3 = pcr(formula1, data = train_data, ncomp = 60, validation= "LOO")
Yhat = predict(model2,test)
Yhat = data.frame(unname(apply(Yhat[,,60], 1, which.max)) - 1)
Y = data.frame(test[,1])
eval_func(unlist(Y),unlist(Yhat))

# Experiment
# 5 percent
split = sample.split(train$V1, 0.05)
train_split = subset(train, split ==T)
X = data.frame(train_split[,-1])
X = scale(X, center = TRUE, scale = FALSE)
y = data.frame(train_split[,1])
ohe_model = OneHotEncoder.fit(y)
y_ohe = as.matrix(transform(ohe_model,y))
y_ohe = as.data.frame(y_ohe)
names(y_ohe) = c("C0","C1","C2","C3","C4","C5","C6","C7","C8","C9")
train_data = data.frame(X,y_ohe)
model4 = pcr(formula1, data = train_data, ncomp = 60)
Y = data.frame(test[,1])
Yhat = predict(model4,test)
Yhat = data.frame(unname(apply(Yhat[,,60], 1, which.max)) - 1)
eval = eval_func(unlist(Y),unlist(Yhat))
eval

# 30 percent
split = sample.split(train$V1, 0.30)
train_split = subset(train, split ==T)
X = data.frame(train_split[,-1])
X = scale(X, center = TRUE, scale = FALSE)
y = data.frame(train_split[,1])
ohe_model = OneHotEncoder.fit(y)
y_ohe = as.matrix(transform(ohe_model,y))
y_ohe = as.data.frame(y_ohe)
names(y_ohe) = c("C0","C1","C2","C3","C4","C5","C6","C7","C8","C9")
train_data = data.frame(X,y_ohe)
model5 = pcr(formula1, data = train_data, ncomp = 60)
Y = data.frame(test[,1])
Yhat = predict(model5,test)
Yhat = data.frame(unname(apply(Yhat[,,60], 1, which.max)) - 1)
eval = eval_func(unlist(Y),unlist(Yhat))
eval

# 70 percent
split = sample.split(train$V1, 0.70)
train_split = subset(train, split ==T)
X = data.frame(train_split[,-1])
X = scale(X, center = TRUE, scale = FALSE)
y = data.frame(train_split[,1])
ohe_model = OneHotEncoder.fit(y)
y_ohe = as.matrix(transform(ohe_model,y))
y_ohe = as.data.frame(y_ohe)
names(y_ohe) = c("C0","C1","C2","C3","C4","C5","C6","C7","C8","C9")
train_data = data.frame(X,y_ohe)
model6 = pcr(formula1, data = train_data, ncomp = 60)
Y = data.frame(test[,1])
Yhat = predict(model6,test)
Yhat = data.frame(unname(apply(Yhat[,,60], 1, which.max)) - 1)
eval = eval_func(unlist(Y),unlist(Yhat))
eval


# Not In scope
# Plot Digits
par(mfrow=c(3,4))
par(mar=c(2,2,2,2))
for(i in 1:12){
  plot_img(train, i)
}

# Percentage of Variance
pvar = cumsum(explvar(model2))
qplot(1:length(pvar),unlist(pvar), 
      xlab="Number of Components", ylab="Variance Explained",
      main="PCR Variance Plot", geom='line')

# Accuracy
qplot(1:pred_dim[3],accuracy_metrics, 
      xlab="Number of Components", ylab="Accuracy",
      main="PCR Accuracy Plot", geom='line') +
  geom_vline(xintercept = line_fix,  color = "red", linetype="dotted")

# PCA 
library(ggfortify)
train$V1 = as.factor(train$V1)
autoplot(prcomp(train[,-1]), data = train, colour = 'V1', label = F,  loadings = F)
