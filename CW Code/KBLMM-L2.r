################################################
## Kernel-Based Learning & Multivariate Modeling
## DMKM-MIRI Masters - September 2016
################################################

set.seed(6046)

################################################
################################################
# EXAMPLE 1: Modelling 2D classification data
################################################
################################################

## the SVM is located in two different packages: one of them is 'e1071'
library(e1071)

## First we create a simple two-class data set:

N <- 200

make.sinusoidals <- function(m,noise=0.2) 
{
  x1 <- c(1:2*m)
  x2 <- c(1:2*m)
  
  for (i in 1:m) {
    x1[i] <- (i/m) * pi
    x2[i] <- sin(x1[i]) + rnorm(1,0,noise)
  }
  
  for (j in 1:m) {
    x1[m+j] <- (j/m + 1/2) * pi
    x2[m+j] <- cos(x1[m+j]) + rnorm(1,0,noise)
  }
  
  target <- c(rep(+1,m),rep(-1,m))
  
  return(data.frame(x1,x2,target))
}

## let's generate the data
dataset <- make.sinusoidals (N)

## and have a look at it
summary(dataset)

plot(dataset$x1,dataset$x2,col=as.factor(dataset$target))

## Now we define a utility function for performing k-fold CV
## the learning data is split into k equal sized parts
## every time, one part goes for validation and k-1 go for building the model (training)
## the final error is the mean prediction error in the validation parts
## Note k=N corresponds to LOOCV

## a typical choice is k=10
k <- 10 
folds <- sample(rep(1:k, length=N), N, replace=FALSE) 

valid.error <- rep(0,k)

C <- 1

## This function is not intended to be useful for general training purposes but it is useful for illustration
## In particular, it does not optimize the value of C (it requires it as parameter)

train.svm.kCV <- function (which.kernel, mycost)
{
  for (i in 1:k) 
  {  
    train <- dataset[folds!=i,] # for building the model (training)
    valid <- dataset[folds==i,] # for prediction (validation)
    
    x_train <- train[,1:2]
    t_train <- train[,3]
    
    switch(which.kernel,
           linear={model <- svm(x_train, t_train, type="C-classification", cost=mycost, kernel="linear", scale = FALSE)},
           poly.2={model <- svm(x_train, t_train, type="C-classification", cost=mycost, kernel="polynomial", degree=2, coef0=1, scale = FALSE)},
           poly.3={model <- svm(x_train, t_train, type="C-classification", cost=mycost, kernel="polynomial", degree=3, coef0=1, scale = FALSE)},
           RBF={model <- svm(x_train, t_train, type="C-classification", cost=mycost, kernel="radial", scale = FALSE)},
           stop("Enter one of 'linear', 'poly.2', 'poly.3', 'radial'"))
    
    x_valid <- valid[,1:2]
    pred <- predict(model,x_valid)
    t_true <- valid[,3]
    
    # compute validation error for part 'i'
    valid.error[i] <- sum(pred != t_true)/length(t_true)
  }
  # return average validation error
  sum(valid.error)/length(valid.error)
}

# Fit an SVM with linear kernel

(VA.error.linear <- train.svm.kCV ("linear", C))

## We should choose the model with the lowest CV error and refit it to the whole learning data
## then use it to predict the test set; we will do this at the end

## As for now we wish to visualize the models

# so first we refit the model:

model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=C, kernel="linear", scale = FALSE)

## Now we are going to visualize what we have done; since we have artificial data, instead of creating
## a random test set, we can create a grid of points as test

plot.prediction <- function (model.name, resol=200)
  # the grid has a (resol x resol) resolution
{
  x <- cbind(dataset$x1,dataset$x2)
  rng <- apply(x,2,range);
  tx <- seq(rng[1,1],rng[2,1],length=resol);
  ty <- seq(rng[1,2],rng[2,2],length=resol);
  pnts <- matrix(nrow=length(tx)*length(ty),ncol=2);
  k <- 1
  for(j in 1:length(ty))
  {
    for(i in 1:length(tx))
    {
      pnts[k,] <- c(tx[i],ty[j])
      k <- k+1
    } 
  }
  
  # we calculate the predictions on the grid
  
  pred <- predict(model, pnts, decision.values = TRUE)
  
  z <- matrix(attr(pred,"decision.values"),nrow=length(tx),ncol=length(ty))
  
  # and plot them
  
  image(tx,ty,z,xlab=model.name,ylab="",axes=FALSE,
        xlim=c(rng[1,1],rng[2,1]),ylim=c(rng[1,2],rng[2,2]),
        col = cm.colors(64))
#        col = rainbow(200, start=0.9, end=0.1))
  
  # then we draw the optimal separation and its margins
  
  contour(tx,ty,z,add=TRUE, drawlabels=TRUE, level=0, lwd=3)
  contour(tx,ty,z,add=TRUE, drawlabels=TRUE, level=1, lty=1, lwd=1, col="grey")
  contour(tx,ty,z,add=TRUE, drawlabels=TRUE, level=-1, lty=1, lwd=1, col="grey")
  
  # then we plot the input data from the two classes
  
  points(dataset[dataset$target==1,1:2],pch=21,col=1,cex=1)
  points(dataset[dataset$target==-1,1:2],pch=19,col=4,cex=1)
  
  # finally we add the SVs
  
  sv <- dataset[c(model$index),];
  sv1 <- sv[sv$target==1,];
  sv2 <- sv[sv$target==-1,];
  points(sv1[,1:2],pch=13,col=1,cex=2)
  points(sv2[,1:2],pch=13,col=4,cex=2)
}

## plot the predictions, the separation, the support vectors, everything
plot.prediction ("linear")

## right, now a quadratic SVM model 

(VA.error.poly.2 <- train.svm.kCV ("poly.2", C))

model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=C, kernel="polynomial", degree=2, coef0=1, scale = FALSE)
plot.prediction ("poly.2")

## right, now a cubic SVM model 

(VA.error.poly.3 <- train.svm.kCV ("poly.3", C))

model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=C, kernel="polynomial", degree=3, coef0=1, scale = FALSE)
plot.prediction ("poly.3")

## and finally an RBF Gaussian SVM model 

(VA.error.RBF <- train.svm.kCV ("RBF", C))

model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=C, kernel="radial", scale = FALSE)
plot.prediction ("RBF")

## Now in a real scenario we should choose the model with the lowest CV error
## which in this case is the RBF

## In a real setting we should optimize the value of C, again with CV; this can be done
## very conveniently using tune() in this package to do automatic grid-search

## another, more general, possibility is to use the train() method in the {caret} package

## Just for illustration, let's see the effect of altering C (significantly):
C <- 50

(VA.error.linear <- train.svm.kCV ("linear", C))
model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=C, kernel="linear", scale = FALSE)
plot.prediction ("linear")

(VA.error.RBF <- train.svm.kCV ("RBF", C))
model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=C, kernel="radial", scale = FALSE)
plot.prediction ("RBF")

C <- 0.05

(VA.error.linear <- train.svm.kCV ("linear", C))
model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=C, kernel="linear", scale = FALSE)
plot.prediction ("linear")

(VA.error.RBF <- train.svm.kCV ("RBF", C))
model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=C, kernel="radial", scale = FALSE)
plot.prediction ("RBF")

################################################
################################################
# EXAMPLE 2: modeling the Promoter Gene data
################################################
################################################

## In genetics, a promoter is a region of DNA that facilitates the transcription of a particular gene. 
## Promoters are located near the genes they regulate.

## Promoter Gene Data: data sample that contains DNA sequences, classified into 'promoters' and 'non-promoters'.
## 106 observations, 2 classes (+ and -)
## The 57 explanatory variables describing the DNA sequence have 4 possible values, represented 
## as the nucleotide at each position:
##    [A] adenine, [C] cytosine, [G] guanine, [T] thymine.

## The goal is to develop a predictive model (a classifier)

################################################
##### data reading
################################################

dd <- read.csv2("promotergene.csv")

p <- ncol(dd)
n <- nrow(dd)

summary(dd)

################################################
# Multiple correspondence analysis
################################################

source ("acm.r")

X <- dd[,2:p]

mc <- acm(X)

plot(mc$rs[,1],mc$rs[,2],type="n")
text(mc$rs[,1],mc$rs[,2],labels=row.names(mc$rs),col=as.numeric(Class))
axis(side=1, pos= 0, labels = F, col="cyan")
axis(side=3, pos= 0, labels = F, col="cyan")
axis(side=2, pos= 0, labels = F, col="cyan")
axis(side=4, pos= 0, labels = F, col="cyan")

# as before, w/o the numbers, for better reading

idict=dd[,1]
plot(mc$rs[,1],mc$rs[,2],col=idict)

# Histogram of eigenvalues

barplot(mc$vaps)

# estimation of the number of dimensions to keep (out of the n-1 = 105)

i <- 1

while (mc$vaps[i] > mean(mc$vaps)) i <- i+1

(nd <- i-1)

################################################
# Logistic Regresion
################################################

# First we create a "maximal" model (using all the predictors)

Psi <- as.data.frame(mc$rs[,1:nd])

attach(dd)

gl1 <- glm(Class~., data=Psi, family=binomial)

summary(gl1)

# it seems the apparent fit is very good, probably due to the large number of regressors

# selection of significant regressors using the AIC

step(gl1)

glf <-  glm(Class ~ V1 + V2 + V22 + V25 + V28 + V38, family = binomial, data = Psi)

summary(glf)

# quality estimation of the model (optimistic!)

glfpred=NULL
glfpred[glf$fitted.values<0.5]=0
glfpred[glf$fitted.values>=0.5]=1
table(dd[,1],glfpred)

# What would be a better way? adjust the model to a learning set and
# use it to predict a test set. Let's see:

# Create a new dataframe 'Psi2' for convenience

Psi2 <- as.data.frame(cbind(Psi,dd[,1]))
names(Psi2)[43] <- "Class"
attach(Psi2)

# split data into learn (2/3) and test (1/3) sets

set.seed(2)
index <- 1:n
learn <- sample(1:n, round(0.67*n))

# refit a new model in the learning (note: there is no need to use
# cross-validation on this set since we use the AIC)

Psi.train <- glm (Class~., data=Psi2[learn,], family=binomial)

# simplify (choose the optimal model)

step(Psi.train)

# note that the results are different ...

Psi.train <- glm (Class~V1 + V2 + V7 + V22 + V38, data=Psi2[learn,], family=binomial)

summary(Psi.train)

# compute new error in 'learn'

glfpred=NULL
glfpred[Psi.train$fitted.values>=0.5]=1
glfpred[Psi.train$fitted.values<0.5]=0

table(dd[learn,1],glfpred)

# again perfect (100%) ... let's see the test ...

Psi.test = predict(Psi.train, newdata=Psi2[-learn,]) 
pt = 1/(1+exp(-Psi.test))

glfpred=NULL
glfpred[Psi.test>=0.5]=1
glfpred[Psi.test<0.5]=0

(tt <- table(dd[-learn,1],glfpred))

# Well, this is more realistic ...

error_rate.test <- 100*(1-sum(diag(tt))/sum(tt))
error_rate.test

# gives a prediction error of 11.4%

################################################
##### Support vector machine
################################################

library(kernlab)

# Note we use LOOCV because the data set is very small; in other circumstances we could use 10CV, 5CV or even better 10x10CV or 10x5CV

# we start with a linear kernel

mi.svm <- ksvm (Class~.,data=Psi2[learn,],kernel='polydot',C=1,cross=length(learn))

# note Number of Support Vectors, Training error and Cross validation error

mi.svm

# choose quadratic kernel first

cuad <- polydot(degree = 2, scale = 1, offset = 1)

mi.svm <- ksvm (Class~.,data=Psi2[learn,],kernel=cuad,C=1,cross=length(learn))

# note Number of Support Vectors, Training error and Cross validation error

mi.svm

# choose now the RBF kernel with automatic adjustment of the variance

mi.svm <- ksvm (Class~.,data=Psi2[learn,],C=1,cross=length(learn))

# note Number of Support Vectors, Training error and Cross validation error

mi.svm

# idem but changing the cost parameter C

mi.svm <- ksvm (Class~.,data=Psi2[learn,],C=5,cross=length(learn))

# note Number of Support Vectors, Training error and Cross validation error

mi.svm

# we choose this latter model and use it to predict the test set

svmpred <- predict (mi.svm, Psi2[-learn,-43])

tt <- table(dd[-learn,1],svmpred)

error_rate.test <- 100*(1-sum(diag(tt))/sum(tt))
error_rate.test

# gives a prediction error of 14.3%

## What if we use the reduced set of variables from logreg?

(mi.svm <- ksvm (Class~V1 + V2 + V7 + V22 + V38,data=Psi2[learn,],C=5,cross=length(learn)))

# It seems clear that we should use this latter model; then

svmpred <- predict (mi.svm, Psi2[-learn,-43])

tt <- table(dd[-learn,1],svmpred)

error_rate.test <- 100*(1-sum(diag(tt))/sum(tt))
error_rate.test

# gives again a prediction error of 11.42%


################################################
## Linear discriminant analysis 
################################################

library(MASS)

mi.lda <- lda (Class ~ ., prior = c(1,1)/2, data = Psi2[learn,], CV=TRUE)
tt <- table(Psi2[learn,43],mi.lda$class)
error_rate.test <- 100*(1-sum(diag(tt))/sum(tt))
error_rate.test

# LOOCV error is 14.1%

mi.lda <- lda (Class ~ ., prior = c(1,1)/2, data = Psi2[learn,], CV=FALSE)
pred <- predict(mi.lda, Psi2[-learn,])$class
tt <- table(Psi2[-learn,43],pred)
error_rate.test <- 100*(1-sum(diag(tt))/sum(tt))
error_rate.test

# test error is 20%

################################################
## Quadratic discriminant analysis
################################################

# can not be used because there are insufficient data, but we can try
# to use it with fewer variables: we choose the logistic regression selection

mi.qda <- qda (Class ~ V1 + V2 + V7 + V22 + V38, prior = c(1,1)/2, data = Psi2[learn,], CV=TRUE)
tt <- table(Psi2[learn,43],mi.qda$class)
error_rate.test <- 100*(1-sum(diag(tt))/sum(tt))
error_rate.test

# LOOCV error is 2.8%

mi.qda <- qda (Class ~ V1 + V2 + V7 + V22 + V38, prior = c(1,1)/2, data = Psi2[learn,], CV=FALSE)
pred <- predict(mi.qda, Psi2[-learn,])$class
tt <- table(Psi2[-learn,43],pred)
error_rate.test <- 100*(1-sum(diag(tt))/sum(tt))
error_rate.test

# we obtain 8.6%, the best so far

# We have reduced the classification error from 50% to 8.6%

################################################
################################################
# EXAMPLE 3: Modelling 1D regression data
################################################
################################################

## A really nice-looking function: 
doppler <- function (x) { sqrt(x*(1-x))*sin(2.1*pi/(x+0.05)) }

N <- 1000

x <- seq(0.2,1,length.out=N)
y <- doppler(x) + rnorm(N,sd=0.1)

# the truth ...

plot(x,xlim=c(0.15,1.0),ylim=c(-0.7,0.7),type="n")
curve (doppler(x), 0.2, 1, add=TRUE, col='magenta')

# the data ...

plot(x,y)

## With this choice of the 'epsilon' and 'gamma' parameters, the SVM underfits the data (blue line) 

model1 <- svm (x,y,epsilon=0.1, type="eps-regression")
lines(x,predict(model1,x),col="blue")

## With this choice of the 'epsilon' and 'gamma' parameters, the SVM overfits the data (green line)

model3 <- svm (x,y,epsilon=0.001,gamma=200, C=100, type="eps-regression")
lines(x,predict(model3,x),col="green")

## With this choice of the 'epsilon' and 'gamma' parameters, the SVM has a very decent fit (red line)
model2 <- svm (x,y,epsilon=0.01,gamma=10, type="eps-regression")
lines(x,predict(model2,x),col="red")


################################################
################################################
# EXAMPLE 4: Modelling 2D Outlier Detection data
################################################
################################################

## just a variation of the built-in example ...

N <- 1000

X <- data.frame(a = rnorm(N), b = rnorm(N))
attach(X)

# default nu = 0.5, to see how it works
(m <- svm(X, gamma = 0.1))

# test:
newdata <- data.frame(a = c(0, 2.5), b = c(0, 2.5))
predict (m, newdata)

# visualize:
plot(X, col = 1:N %in% m$index + 1, xlim = c(-5,5), ylim=c(-5,5))
points(newdata, pch = "+", col = 2, cex = 5)

# now redo with nu = 0.01 (more in accordance with outlier detection)

(m <- svm(X, gamma = 0.1, nu = 0.01))

plot(X, col = 1:N %in% m$index + 1, xlim = c(-5,5), ylim=c(-5,5))
points(newdata, pch = "+", col = 2, cex = 5)

