################################################
## Kernel-Based Learning & Multivariate Modeling
## DMKM-MIRI Masters - November 2016
################################################

dd <- read.table ("biosensor_data.csv", header=FALSE, sep=",", dec=".")
colnames(dd) <- c("Glucose", "Benzoquinone", "T", "PH", "Target")
summary(dd)

n <- nrow(dd)

dd[,-5] <- scale(dd[,-5])

set.seed (104)
dd <- dd[sample.int(n),]

##### First basic analysis:

summary(dd)

boxplot(dd[,1:4])

par (mfrow=c(1,2))
boxplot(dd[,5], main="Before taking log")
boxplot(log(dd[,5]), main="After taking log")
par (mfrow=c(1,1))

dd$Target <- log(dd$Target)


library(kernlab)

# learn-test splitting & perform TIMESxCV cross-validation on the learning part

nl <- 220 # 68.75% for TR, 31.25% for TE

learn <- sample(1:n, nl)

# SVM-R
# 
# linear kernel

svm1 <- ksvm (Target ~ ., data=dd[learn,], cross=10, epsilon=0.1, kernel='polydot', kpar=list(degree=1, scale=1, offset=1))

pred <- predict(svm1, newdata=dd[learn,])
(R2 <- 1 - sum((dd[learn,]$Target - pred)^2) / (var(dd[learn,]$Target)*(n-1)))

plot(dd[learn,]$Target, pred, pch=20, col="gray", xlab="truth", ylab="prediction")

# cubic kernel

svm3 <- ksvm (Target ~ ., data=dd[learn,], cross=10, epsilon=0.1, kernel='polydot', kpar=list(degree=3, scale=1, offset=1))

pred <- predict(svm3, newdata=dd[learn,])
(R2 <- 1 - sum((dd[learn,]$Target - pred)^2) / (var(dd[learn,]$Target)*(n-1)))

plot(dd[learn,]$Target, pred, pch=20, col="gray", xlab="truth", ylab="prediction")


# RBF kernel

svmR <- ksvm (Target ~ ., data=dd[learn,], cross=10, epsilon=0.1)

pred <- predict(svmR, newdata=dd[learn,])
(R2 <- 1 - sum((dd[learn,]$Target - pred)^2) / (var(dd[learn,]$Target)*(n-1)))

# here we should proceed with the tuning of the SVM-R parameters (C,epsilon) using CV in a TR/TE split setting

# RVM

rvm1 <- rvm (Target ~ ., data=dd[learn,], cross=10, kernel='polydot', kpar=list(degree=1, scale=1, offset=1))

rvm3 <- rvm (Target ~ ., data=dd[learn,], cross=10, kernel='polydot', kpar=list(degree=3, scale=1, offset=1))

rvmR <- rvm (Target ~ ., data=dd[learn,], cross=10)

# print relevance vectors
alpha(rvm3)
RVindex(rvm3)

# here we go with some tuning of both the SVM and RVM

library(ipred)

# Resampling control: TIMESxK-cross validation (stratified)

K <- 10
TIMES <- 10
# here we go

# note we should optimize (C, epsilon) for the SVM!
mean(replicate(TIMES, cross(ksvm (Target ~ ., data=dd[learn,], kernel=polydot, kpar=list(degree=3, scale=1, offset=1), cross=K))))

mean(replicate(TIMES, cross(rvm (Target ~ ., data=dd[learn,], kernel=polydot, kpar=list(degree=3, scale=1, offset=1), cross=K))))
  
# prediction of the held out test:

# 1. refit with best parameters on training data

final.SVM <- ksvm (Target ~ ., data=dd[learn,], kernel=polydot, kpar=list(degree=3, scale=1, offset=1), cross=0)

final.RVM <- rvm (Target ~ ., data=dd[learn,], kernel=polydot, kpar=list(degree=3, scale=1, offset=1), cross=0)

# 2. predict test data

SVM.pred <- predict (final.SVM, newdata=dd[-learn,])

(1 - sum( (dd$Target[-learn] - SVM.pred)^2 ) / ( var(dd$Target[-learn])*(n-nl-1) ))

RVM.pred <- predict (final.RVM, newdata=dd[-learn,])

(1 - sum( (dd$Target[-learn] - RVM.pred)^2 ) / ( var(dd$Target[-learn])*(n-nl-1) ))
