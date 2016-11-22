# http://archive.ics.uci.edu/ml/datasets/Ionosphere

# Attribute Information:     
# All 34 are continuous, as described above
# The 35th attribute is either "good" or "bad" according to the definition
# summarized above.  This is a binary classification task.
rm(list=ls())
library(kernlab)
library(caTools)
library(cvTools)
library(e1071)
library(MASS)
library(caret)
library(doParallel)
library(dplyr)
library(tidyr)
library(ggplot2)
set.seed(825)

data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
ion_data = read.csv(url(data_url), header = FALSE)
ion_data = ion_data[sample(nrow(ion_data), nrow(ion_data)), ]
head(ion_data)


# Removing column V2 as it contains only zeros
ion_data = subset(ion_data, select = -V2)
str(ion_data)
k_fold = 5


split = sample.split(ion_data$V35, SplitRatio = 0.8)
train_df = subset(ion_data, split == TRUE)
# We will assume this as our test data and ignore the labels for now
test_df = subset(ion_data, split == FALSE)

folds = cvFolds(NROW(train_df), K=k_fold)
formual =as.formula(V35 ~ .)

# Evaluation Function
eval = function(y,y_hat,fold_name, time){
  cm = table(y, y_hat)
  accuracy = sum(diag(cm))/sum(cm)
  
  TN = cm[1,1]
  TP = cm[2,2]
  FN = cm[2,1]
  FP = cm[1,2]
  f1=(2*TP)/(2*TP+FP+FN)
  data.frame(fold_name,accuracy,f1,time)
}

nb_eval = data.frame()
for(i_loop in 1:k_fold){
  train = train_df[folds$subsets[folds$which != i_loop], ]
  val = train_df[folds$subsets[folds$which == i_loop], ]
  st =  Sys.time()
  model = caret::train(formual,data=train,method="nb")
  pred = predict(model,newdata=val)
  et = Sys.time()
  tt = as.numeric(et - st)
  fold_model_name = paste('fold_',i_loop,sep="")
  nb_eval = rbind(nb_eval,eval(val$V35, pred,fold_model_name,tt))
}
nb_eval


rpart_eval = data.frame()
for(i_loop in 1:k_fold){
  train = train_df[folds$subsets[folds$which != i_loop], ]
  val = train_df[folds$subsets[folds$which == i_loop], ]
  st =  Sys.time()
  model = caret::train(V35 ~ .,data=ion_data, method="rpart")
  pred = predict(model,newdata=val)
  et = Sys.time()
  tt =  as.numeric(et - st)
  fold_model_name = paste('fold_',i_loop,sep="")
  rpart_eval = rbind(rpart_eval,eval(val$V35, pred,fold_model_name,tt))
}
rpart_eval

rf_eval = data.frame()
for(i_loop in 1:k_fold){
  train = train_df[folds$subsets[folds$which != i_loop], ]
  val = train_df[folds$subsets[folds$which == i_loop], ]
  st =  Sys.time()
  model = caret::train(formual,data=train,method="rf")
  pred = predict(model,newdata=val)
  et = Sys.time()
  tt = as.numeric(et - st)
  fold_model_name = paste('fold_',i_loop,sep="")
  rf_eval = rbind(rf_eval,eval(val$V35, pred,fold_model_name,tt))
}
rf_eval

nn_eval = data.frame()
for(i_loop in 1:k_fold){
  train = train_df[folds$subsets[folds$which != i_loop], ]
  val = train_df[folds$subsets[folds$which == i_loop], ]
  st =  Sys.time()
  model = caret::train(formual,data=train,method="nnet")
  pred = predict(model,newdata=val)
  et = Sys.time()
  tt = as.numeric(et - st)
  fold_model_name = paste('fold_',i_loop,sep="")
  nn_eval = rbind(nn_eval,eval(val$V35, pred,fold_model_name,tt))
}

lda_eval = data.frame()
for(i_loop in 1:k_fold){
  train = train_df[folds$subsets[folds$which != i_loop], ]
  val = train_df[folds$subsets[folds$which == i_loop], ]
  st =  Sys.time()
  model = caret::train(formual,data=train,method="lda")
  pred = predict(model,newdata=val)
  et = Sys.time()
  tt = as.numeric(et - st)
  fold_model_name = paste('fold_',i_loop,sep="")
  lda_eval = rbind(lda_eval,eval(val$V35, pred,fold_model_name,tt))
}
lda_eval

gbm_eval = data.frame()
for(i_loop in 1:k_fold){
  train = train_df[folds$subsets[folds$which != i_loop], ]
  val = train_df[folds$subsets[folds$which == i_loop], ]
  st =  Sys.time()
  model = caret::train(formual,data=train,method="gbm")
  pred = predict(model,newdata=val)
  et = Sys.time()
  tt = as.numeric(et - st)
  fold_model_name = paste('fold_',i_loop,sep="")
  gbm_eval = rbind(gbm_eval,eval(val$V35, pred,fold_model_name,tt))
}
gbm_eval

ksvm1_eval = data.frame()
for(i_loop in 1:k_fold){
  train = train_df[folds$subsets[folds$which != i_loop], ]
  val = train_df[folds$subsets[folds$which == i_loop], ]
  st =  Sys.time()
  model = ksvm(formual, data=train, kernel='rbfdot',type='nu-svc')
  pred = predict(model,newdata=val)
  et = Sys.time()
  tt = as.numeric(et - st)
  fold_model_name = paste('fold_',i_loop,sep="")
  ksvm1_eval = rbind(ksvm1_eval,eval(val$V35, pred,fold_model_name,tt))
}
mean(ksvm1_eval$f1)

ksvm2_eval = data.frame()
for(i_loop in 1:k_fold){
  train = train_df[folds$subsets[folds$which != i_loop], ]
  val = train_df[folds$subsets[folds$which == i_loop], ]
  st =  Sys.time()
  model = ksvm(formual, data=train, kernel='vanilladot',type='nu-svc')
  pred = predict(model,newdata=val)
  et = Sys.time()
  tt = as.numeric(et - st)
  fold_model_name = paste('fold_',i_loop,sep="")
  ksvm2_eval = rbind(ksvm2_eval,eval(val$V35, pred,fold_model_name,tt))
}

ksvm3_eval = data.frame()
for(i_loop in 1:k_fold){
  train = train_df[folds$subsets[folds$which != i_loop], ]
  val = train_df[folds$subsets[folds$which == i_loop], ]
  st =  Sys.time()
  
  model = ksvm(formual, data=train, kernel='poly',type='nu-svc')
  pred = predict(model,newdata=val)
  et = Sys.time()
  tt = as.numeric(et - st)
  fold_model_name = paste('fold_',i_loop,sep="")
  ksvm3_eval = rbind(ksvm3_eval,eval(val$V35, pred,fold_model_name,tt))
}

# Color Pallet
tol9qualitative = c("#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499")

# F1 Score
f1 = data.frame(cbind(nb_eval$f1,rpart_eval$f1,
                      rf_eval$f1,nn_eval$f1,lda_eval$f1,gbm_eval$f1,
                      ksvm1_eval$f1,ksvm2_eval$f1,ksvm3_eval$f1))
names(f1) = c('nb','rpart','rf','nn','lda','gbm','rbfk','dotk','polyk')
boxplot(f1,
        main = "Model", ylab = "F1 Score", col = tol9qualitative,
        las=2,ylim = c(0.83,1.0))
colMeans(f1)


# Accuracy
acc = data.frame(cbind(nb_eval$accuracy,rpart_eval$accuracy,
                      rf_eval$accuracy,nn_eval$accuracy,
                      lda_eval$accuracy,gbm_eval$accuracy,
                      ksvm1_eval$accuracy,ksvm2_eval$accuracy,
                      ksvm3_eval$accuracy))

names(acc) = c('nb','rpart','rf','nn','lda','gbm','rbfk','dotk','polyk')

boxplot(acc,
        main = "Models", ylab = "Accuracy", col = tol9qualitative,
        las=2,ylim = c(0.80,1.0))

print("Average Accuracy over 5 Folds per model")
colMeans(acc)

# Time
time_taken = data.frame(cbind(nb_eval$time,rpart_eval$time,
                              rf_eval$time,nn_eval$time,
                              lda_eval$time,gbm_eval$time,
                              ksvm1_eval$time,ksvm2_eval$time,
                              ksvm3_eval$time))
names(time_taken) =  c('nb','rpart','rf','nn','lda','gbm','rbfk','dotk','polyk')
boxplot(time_taken,
        main = "Models", ylab = "Seconds", col = tol9qualitative,
        las=2,ylim = c(0.3,20))
print("Average Time Taken per model")
colMeans(time_taken)


# Hyper Parameter Tuning
sigma1 = c(0.001,0.01,0.01,0.1)
for(sigma in sigma1) {
  ksvmt_eval = data.frame()
  for(i_loop in 1:k_fold){
    train = train_df[folds$subsets[folds$which != i_loop], ]
    val = train_df[folds$subsets[folds$which == i_loop], ]
    st =  Sys.time()
    formual =as.formula(V35 ~ .)
    rbf2 <- rbfdot(sigma=sigma)
    model = ksvm(formual, data=train, kernel=rbf2,type='nu-svc', C=1)
    pred = predict(model,newdata=val)
    et = Sys.time()
    tt = as.numeric(et - st)
    fold_model_name = paste('fold_',i_loop,sep="")
    ksvmt_eval = rbind(ksvmt_eval,eval(val$V35, pred,fold_model_name,tt))
  }
  print(mean(ksvmt_eval$f1,sigma))
}

# Finally train RBF Kernel with whole training data and Evaluate on test set
st =  Sys.time()
rbf3 <- rbfdot(sigma=0.1)
model = ksvm(formual, data=train_df, kernel=rbf3,type='nu-svc', C=1)
pred = predict(model,newdata=test_df)
et = Sys.time()
tt = as.numeric(et - st)
eval(test_df$V35, pred,'Final Evaluation',tt)

