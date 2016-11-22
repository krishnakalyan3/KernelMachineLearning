####################################################################
# KBLMM - DMKM/MIRI Masters
# Lluís A. Belanche

# LAB 1: Kernel ridge regression
# version of September 2015
####################################################################

set.seed(6046)

## We first show standard (non-ridge) regression and then kernel (ridge) regression, as seen in class

###########################
## a really nice-looking function

A <- 20
x <- seq(-A,A,by=0.11)
t <- sin(x)/x + rnorm(x,sd=0.03)

plot(x,t,type="l")

###########################
## standard (non-ridge) regression

d <- data.frame(x,t)

linreg.1 <- lm (d)

plot(x,t,type="l")

abline(linreg.1, col="yellow")

## the result is obviously terrible, because our function is far from linear

## suppose we use a quadratic polynomial now:

linreg.2 <- lm (t ~ x + I(x^2), d)

plot(x,t,type="l")

points(x, predict(linreg.2), col="red", type="l")

## and keep increasing the degree ... in the end we would certainly do polynomial regression (say, degree 6):

linreg.6 <- lm (t ~ poly(x,6), d)

points(x, predict(linreg.6), col="green", type="l")

## and keep increasing the degree ... 

linreg.11 <- lm (t ~ poly(x,11), d)

points(x, predict(linreg.11), col="blue", type="l")

## we get something now ... but wait: instead of extracting new features manually (the higher order monomials), it is much better to use a kernel function: let's do kernel (ridge) regression with the RBF kernel

## the reason for using regularization is that with the RBF kernel we extract monomials to infinite degrees
## so we need to explicitly control the complexity

###########################
## kernel ridge regression

## Let's start by computing the Gaussian RBF kernel manually

N <- length(x)
sigma <- 1
kk <- tcrossprod(x)
dd <- diag(kk)

## note that 'crossprod' and 'tcrossprod' are simply matrix multiplications (i.e., dot products)
## see help(crossprod) for details

## crossprod is a function of two arguments x,y; if only one is given, the second is taken to be the same as the first

## this computes the RBF kernel rather quickly
myRBF.kernel <- exp(sigma*(-matrix(dd,N,N)-t(matrix(dd,N,N))+2*kk))

dim(myRBF.kernel)

## the first 5 entries (note diagonal is always 1)
myRBF.kernel[1:5,1:5]

## this is a good moment to review the class notes about kernel (ridge) regression

lambda <- 0.01

ident.N <- diag(rep(1,N))

alphas <- solve(myRBF.kernel + lambda*ident.N)

alphas <- alphas %*% t

lines(x,myRBF.kernel %*% alphas,col="magenta")

## not bad, a little bit wiggly, but essentially OK. The important point is that we have converted a linear technique
## into a non-linear one, by introducing the kernel

## if we add more regularization:

lambda <- 1

alphas <- solve(myRBF.kernel + lambda*ident.N)

alphas <- alphas %*% t

plot(x,t,type="l")
lines(x,myRBF.kernel %*% alphas,col="red")

## that is it.
## 
## It has to be said that the RBF kernel has a tunable parameter (sigma), which we left constant in this little exercise.
## As an indication, increasing sigma leads to a better fit to training data (danger of ending up overfitting), and decreasing sigma (towards zero) leads to a worse fit to training data (danger of ending up underfitting).
## 
## On the downside, the need of inverting a large matrix can be a big issue as N grows. Clearly we need better methods ... (although notice that the kernel matrix does not depend on data dimension)
