rm(list=ls())
library(doMC)
library(dplyr)
library(pls)
library(plsdepot)
registerDoMC(8)

setwd('/Users/krishna/MIRI/MVA/Octane_PLS/')
source('targets_prep.R')

# Preperare Training Data
train = read.csv('exer_leukemia/data_set_ALL_AML_train.csv', sep=';')
train_num = select(train,starts_with("X"))
train_t = data.frame(t(train_num))
rownames(train_t) = as.integer(substring(rownames(train_t),2))
train_prep = train_t[order(as.integer(rownames(train_t))),]
train_prep_center = data.frame(scale(train_prep, center = T, scale = F) )
train_prep_center$target = target_train

# Prepare Test data
test = read.csv('exer_leukemia/data_set_ALL_AML_independent.csv', sep=';')
test_num = select(test,starts_with("X"))
test_t = data.frame(t(test_num))
rownames(test_t) = as.integer(substring(rownames(test_t),2))
test_prep = test_t[order(as.integer(rownames(test_t))),]
test_prep_center = data.frame(scale(test_prep, 
                                    center = colMeans(train_prep), 
                                    scale = F))

test_prep_center$target = target_test

# Model
model1 = plsr(target ~ ., data=train_prep_center, validation = "LOO")

plot(R2(model1), main = 'R^2 vs Number of Components', 
     xlab = 'Number of Components', ylab = 'R^2')
abline(v=5, cex=0.9,col='red',lty = 2)

# Test data Projection
test_projection = as.matrix(test_prep_center[,-c(7130)]) %*% model1$projection[,1:5]
plot(model1$scores[,1:2], col = as.factor(target_train))
points(test_projection[,1:2], pch=19, col = as.factor(target_test))

source('Util.R')

# Evaluation
yhat_pred = predict(model1,test_prep_center)
yhat_comp = data.frame(yhat_pred)
yhat = yhat_comp[,5] >= .5
y = test_prep_center$target
eval_pls = eval_func(y,yhat, cm_show = T)

# Logist Regression Model
train_glm = data.frame(cbind(model1$scores[,1:5],target_train))
names(train_glm) = c(paste0('C.',rep(1:5)),'target')

train_glm$target = as.factor(train_glm$target)
test_glm = as.data.frame(test_projection)
names(test_glm) = paste0('C.',rep(1:5))
model2 = glm(target ~. , data=train_glm, family="binomial")
summary(model2)
yhat = predict(model2, test_glm, type='response') > .5
eval_glm = eval_func(y,yhat, cm_show = T)


test_projection = as.matrix(test_prep_center[,-c(7130)]) %*% model1$projection[,1:5]
plot(model1$scores[,1:2], col = as.factor(target_train))
points(test_projection[,1:2], pch=19, col = as.factor(target_test))
points(test_projection[5,1],test_projection[5,2],pch=19, col = 'blue')