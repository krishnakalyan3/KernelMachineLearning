################################################
## Kernel-Based Learning & Multivariate Modeling
## DMKM-MIRI Masters - October 2016
################################################

set.seed(6046)
library(kernlab)

#################################################
# SVM for 2D Gaussian data with hand-made kernels
#################################################

# This initial code is heavily based on J.P. Vert's excelllent teaching material

# Original file is available at
#   http://cbio.ensmp.fr/~jvert/svn/tutorials/practical/makekernel/makekernel.R
# The corresponding note is available at
#   http://cbio.ensmp.fr/~jvert/svn/tutorials/practical/makekernel/makekernel_notes.pdf

## First we create a simple two-class data set:

N <- 200 # number of data points
d <- 2   # dimension
sigma <- 2  # variance of the distribution
meanpos <- 0 # centre of the distribution of positive examples
meanneg <- 3 # centre of the distribution of negative examples
npos <- round(N/2) # number of positive examples
nneg <- N-npos # number of negative examples

## Generate the positive and negative examples
xpos <- matrix(rnorm(npos*d,mean=meanpos,sd=sigma),npos,d)
xneg <- matrix(rnorm(nneg*d,mean=meanneg,sd=sigma),npos,d)
x <- rbind(xpos,xneg)

## Generate the class labels
t <- matrix(c(rep(1,npos),rep(-1,nneg)))

## Visualize the data
plot(x,col=ifelse(t>0,1,2))
legend("topleft",c('Pos','Neg'),col=seq(2),pch=1,text.col=seq(2))

## Now let's train a SVM with the standard (built-in) RBF kernel
## see help(kernels) for definition of this and other built-in kernels

## a) Let's start by computing the Gaussian RBF kernel manually
sigma <- 1
kk <- tcrossprod(x)
dd <- diag(kk)

## note that 'crossprod' and 'tcrossprod' are simply matrix multiplications (i.e., dot products)
## see help(crossprod) for details
## it is a function of two arguments x,y; if only one is given, the second is taken to be the same as the first

## make sure you understand why this computes the RBF kernel rather quickly
myRBF.kernel <- exp(sigma*(-matrix(dd,N,N)-t(matrix(dd,N,N))+2*kk))

dim(myRBF.kernel)

## the first 5 entries (note diagonal is always 1)
myRBF.kernel[1:5,1:5]

## Now we would like to train a SVM with our precomputed kernel

## We basically have two options in {kernlab}:

## either we explicitly convert myRBF.kernel to a 'kernelMatrix' object, and then ksvm() understands it
svm1 <- ksvm (as.kernelMatrix(myRBF.kernel), t, type="C-svc")

## or we keep it as a regular matrix and we add the kernel='matrix' argument
svm2 <- ksvm(myRBF.kernel,t, type="C-svc", kernel='matrix')

## b) This is how we would do it "the classical way" (since this is a built-in kernel)

## kpar is the way to pass parameters to a kernel (called kernel parameters)
## WARNING: the ksvm() method scales the data by default; to prevent it, use scale=c()

svm3 <- ksvm(x,t, type="C-svc", kernel='rbf', kpar=list(sigma=1),scale=c())

## Now we compare the 3 formulations, they *should* be exactly the same

## Note also that the built-in version is faster (it is written in C code)

svm1
svm2
svm3

## Now we are going to make predictions with our hand-computed kernel

## First we split the data into training set and test set
ntrain <- round(N*0.8)     # number of training examples
tindex <- sample(N,ntrain) # indices of training samples

## Then we train the svm with our kernel over the training points
svm1.train <- ksvm (myRBF.kernel[tindex,tindex],t[tindex], type="C-svc",kernel='matrix')

## Let's call SV the set of obtained support vectors

## Then it becomes tricky. We must compute the test-vs-SV kernel matrix
## which we do in two phases:

# First the test-vs-train matrix
testK <- myRBF.kernel[-tindex,tindex]
# then we extract the SV from the train
testK <- testK[,SVindex(svm1.train),drop=FALSE]

# Now we can predict the test data
# Warning: here we MUST convert the matrix testK to a 'kernelMatrix'
y1 <- predict(svm1.train,as.kernelMatrix(testK))

# Do the same with the usual built-in kernel formulation
svm2.train <- ksvm(x[tindex,],t[tindex], type='C-svc', kernel='rbf', kpar=list(sigma=1), scale=c())

y2 <- predict(svm2.train,x[-tindex,])

# Check that the predictions are the same
table(y1,y2)

# Check the real performance
table(y1, t[-tindex])
cat('Error rate = ',100*sum(y1!=t[-tindex])/length(y1),'%')


################################################
# SVM for classification with a string kernel
################################################

## We are going to use a slightly-processed version of the famous
## Reuters news articles dataset.  All articles with no Topic
## annotations are dropped. The text of each article is converted to
## lowercase, whitespace is normalized to single-spaces.  Only the
## first term from the Topic annotation list is retained (some
## articles have several topics assigned).  

## The resulting dataset is a list of pairs (Topic, News content) We willl only use three topics for analysis: Crude Oil, Coffee and Grain-related news

## The resulting data frame contains 994 news items on crude oil,
## coffee and grain. The news text is the column "Content" and its
## category is the column "Topic". The goal is to create a classifier
## for the news articles.

## Note that we can directly read the compressed version (reuters.txt.gz). 
## There is no need to unpack the gz file; for local files R handles unpacking automagically

reuters <- read.table("reuters.txt.gz", header=T)

# We leave only three topics for analysis: Crude Oil, Coffee and Grain-related news
reuters <- reuters[reuters$Topic == "crude" | reuters$Topic == "grain" | reuters$Topic == "coffee",]

reuters$Content <- as.character(reuters$Content)    # R originally loads this as factor, so needs fixing
reuters$Topic <- factor(reuters$Topic)              # re-level the factor to have only three levels

levels(reuters$Topic)

length(reuters$Topic)

table(reuters$Topic)

## an example of a text about coffee
reuters[2,]

## an example of a text about grain
reuters[7,]

## an example of a text about crude oil
reuters[12,]

(N <- dim(reuters)[1])  # number of rows

# we shuffle the data first
set.seed(12)
reuters <- reuters[sample(1:N, N),]

# To deal with textual data we need to use a string kernel. Several such kernels are implemented in the "stringdot" method of the kernlab package. We shall use the simplest one: the p-spectrum kernel. The feature map represents the string as a multiset of its substrings of length p

# Example, for p=2 we have

# phi("ababc") = ("ab" -> 2, "ba" -> 1, "bc" --> 1, other -> 0)

# we can define a normalized 3-spectrum kernel (p is length)
k <- stringdot("spectrum", length=3, normalized=T)

# Let's see some examples:

k("I did it my way", "I did it my way")

k("He did it his way", "I did it my way")

k("I did it my way", "She did it her way")

k("I did it my way", "Let's get our way out")

## We start by doing a kPCA (we'll see this in the next lecture)

## first we define a modified plotting function 

plotting <-function (kernelfu, kerneln)
{
  xpercent <- eig(kernelfu)[1]/sum(eig(kernelfu))*100
  ypercent <- eig(kernelfu)[2]/sum(eig(kernelfu))*100
  
  plot(rotated(kernelfu), col=as.integer(reuters$Topic),
       main=paste(paste("Kernel PCA (", kerneln, ")", format(xpercent+ypercent,digits=3)), "%"),
       xlab=paste("1st PC -", format(xpercent,digits=3), "%"),
       ylab=paste("2nd PC -", format(ypercent,digits=3), "%"))
}

## Create a kernel matrix using 'k' as kernel

k <- stringdot("spectrum", length=5, normalized=T)
K <- kernelMatrix(k, reuters$Content)
dim(K)

K[2,2]

K[2,3:10]

## Plot the result using the first 2 PCs (we can add colors for the two classes)

kpc.reuters <- kpca (K, features=2, kernel="matrix")
plotting (kpc.reuters,"5 - spectrum kernel")

## finally add a legend
legend("topleft", legend=c("crude oil", "coffee","grain"),    
       pch=c(1,1),                    # gives appropriate symbols
       col=c("red","black", "green")) # gives the correct color

## We can also train a SVM using this kernel matrix in the training set

## First we should split the data into learning (2/3) and test (1/3) parts
ntrain <- round(N*2/3)     # number of training examples
tindex <- sample(N,ntrain) # indices of training examples
  
## The fit a SVM in the train part
svm1.train <- ksvm (K[tindex,tindex],reuters$Topic[tindex], type="C-svc", kernel='matrix')

## and make it predict the test part

## Let's call SV the set of obtained support vectors

## Then it becomes tricky. We must compute the test-vs-SV kernel matrix
## which we do in two phases:

# First the test-vs-train matrix
testK <- K[-tindex,tindex]
# then we extract the SV from the train
testK <- testK[,SVindex(svm1.train),drop=FALSE]

# Now we can predict the test data
# Warning: here we MUST convert the matrix testK to a 'kernelMatrix'
y1 <- predict(svm1.train,as.kernelMatrix(testK))

table (pred=y1, truth=reuters$Topic[-tindex])

cat('Error rate = ',100*sum(y1!=reuters$Topic[-tindex])/length(y1),'%')

## now we define a 3D plotting function

library("rgl")
open3d()

plotting3D <-function (kernelfu, kerneln)
{
  xpercent <- eig(kernelfu)[1]/sum(eig(kernelfu))*100
  ypercent <- eig(kernelfu)[2]/sum(eig(kernelfu))*100
  zpercent <- eig(kernelfu)[3]/sum(eig(kernelfu))*100
  
  # resize window
  par3d(windowRect = c(100, 100, 612, 612))
  
  plot3d(rotated(kernelfu), 
         col  = as.integer(reuters$Topic),
         xlab = paste("1st PC -", format(xpercent,digits=3), "%"),
         ylab = paste("2nd PC -", format(ypercent,digits=3), "%"),
         zlab = paste("3rd PC -", format(zpercent,digits=3), "%"),
         main = paste("Kernel PCA"), 
         sub = "red - crude oil | black - coffee | green - grain",
         top = TRUE, aspect = FALSE, expand = 1.03)
 }

kpc.reuters <- kpca (K, features=3, kernel="matrix")
plotting3D (kpc.reuters,"5 - spectrum kernel")

#################################################
# Creating our own hand-made kernels
#################################################

## Now we are going to understand better the class 'kernel' in kernlab

## An object of class 'kernel' is simply a function with an additional slot 'kpar' for kernel parameters

## We can start by looking at two built-in kernels to see how they were created
vanilladot
rbfdot

## Let us create a RBF kernel and look at its attributes
rbf <- rbfdot(sigma=1)
rbf
rbf@.Data # the kernel function itself
rbf@kpar  # the kernel paramters
rbf@class # the class

## Once we have a kernel object such as rbf, we can do several things, eg:

## 1) Compute the kernel between two vectors
rbf(x[1,],x[2,])

## 2) Compute a kernel matrix between two sets of vectors
K <- kernelMatrix(rbf,x[1:5,],x[6:20,])
dim(K)

## or between a set of vectors with itself (this is the typical use)
K <- kernelMatrix(rbf,x)
dim(K)

## 3) Obviously we can train a SVM
m <- ksvm (x,t, kernel=rbf, type="C-svc", scale=c())

## Now we are going to make our own kernel and integrate it in kernlab: 

## To make things simple, we start with our "own version" of the linear kernel

kval <- function(x, y = NULL) 
{
  if (is.null(y)) {
    crossprod(x)
  } else {
    crossprod(x,y)
  }
}

## We then create the kernel object as follows
## Remember this kernel has no parameters, so we specify kpar=list(), an empty list

mylinearK <- new("kernel",.Data=kval,kpar=list())

## this is what we did
str(mylinearK)

## Now we can call different functions of kernlab right away

mylinearK (x[1,],x[2,])

kernelMatrix (mylinearK,x[1:5,])

m <- ksvm(x,t, kernel=mylinearK, type="C-svc")

# Check that we get the same results as the normal vanilla kernel (linear kernel)
linearK <- vanilladot()
linearK(x[1,],x[2,])
kernelMatrix(linearK,x[1:5,])
m <- ksvm(x,t, kernel=linearK, type="C-svc")

## As a final example, we make a kernel that evaluates a precomputed kernel
## This is particularly useful when the kernel is very costly to evaluate
## so we do it once and store in a external file, for example

## The way we do this is by creating a new "kernel" whose parameter is a precomputed kernel matrix K
## The kernel function is then a function of integers i,j such that preK(i,j)=K[i,j]

mypreK <- function (preK=matrix())
{
  rval <- function(i, j = NULL) {
    ## i, j are just indices to be evaluated
    if (is.null(j)) 
    {
      preK[i,i]
    } else 
    {
      preK[i,j]
    }
  }
  return(new("kernel", .Data=rval, kpar=list(preK = preK)))
}

## To simplify matters, suppose we already loaded the kernel matrix from disk into
## our matrix 'myRBF.kernel' (the one we created at the start)

## We create it
myprecomputed.kernel <- mypreK(myRBF.kernel)
str(myprecomputed.kernel)

## We check that it works

myRBF.kernel[seq(5),seq(5)]                 # original matrix (seen just as a matrix)
kernelMatrix(myprecomputed.kernel,seq(5))   # our kernel

## We can of course use it to train SVMs

svm.pre <- ksvm(seq(N),t, type="C-svc", kernel=myprecomputed.kernel, scale=c())
svm.pre

## which should be equal to our initial 'svm1'
svm1

## compare the predictions are equal
p1 <- predict (svm.pre, seq(N))
p2 <- predict (svm1)[1:N]
table(p1,p2)

## that is all, folks ...