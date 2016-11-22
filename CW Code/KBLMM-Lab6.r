################################################
## Kernel-Based Learning & Multivariate Modeling
## DMKM-MIRI Masters - October 2016
################################################

## We are now going to use the {kernlab} package
## In order to get the maximum of this nice package, I recommend that you have a look
## at the manual http://cran.r-project.org/web/packages/kernlab/kernlab.pdf

library(kernlab)
set.seed(6046)

################################################
################################################
# EXAMPLE 1: PCA and KPCA: a simple starter
################################################
################################################

# Let's create simple string data

subjects <-
      as.matrix(c("Philosophy","Geography","Biology","Statistics","BioStatistics",
      "Chemistry","Geology","Genealogy","Informatics","BioInformatics",
      "Physics","BioTechnology","Mathematics","Gardening"))

subjects.kpca <- kpca (subjects,  kernel = "stringdot", kpar = list(length = 4))


## first we create a plotting function 

plotting <-function (kernelfu, kerneln)
{
  xpercent <- eig(kernelfu)[1]/sum(eig(kernelfu))*100
  ypercent <- eig(kernelfu)[2]/sum(eig(kernelfu))*100
  
  plot(rotated(kernelfu), 
       main=paste(paste("Kernel PCA (", kerneln, ")", format(xpercent+ypercent,digits=3)), "%"),
       xlab=paste("1st PC -", format(xpercent,digits=3), "%"),
       ylab=paste("2nd PC -", format(ypercent,digits=3), "%"))
  
  text(jitter(rotated(kernelfu)[,1],amount=0.1), jitter(rotated(kernelfu)[,2],amount=0.05), subjects, cex=0.75, pos=3)
}

plotting (subjects.kpca, "4-stringdot")


################################################
################################################
# EXAMPLE 2: PCA and KPCA in action
################################################
################################################


## We want to perform a principal components analysis on the USArrests dataset, which
## contains Lawyers' ratings of 43 state judges in the US Superior Court.

## see USJudgeRatings {datasets}

summary(USJudgeRatings)
?USJudgeRatings
USJudgeRatings

## Have a look at the data; it seems that most ratings are highly correlated

require(graphics)
pairs(USJudgeRatings, panel = panel.smooth, main = "US Judge Ratings data")

require(psych)
describe (USJudgeRatings)

## the variances (sd²) of the variables do not vary much, but scaling is many times appropriate
## there is no need to do it beforehand, since prcomp() is able to handle it

## First we perform good old standard PCA

pca.JR <- prcomp(USJudgeRatings, scale = TRUE)
pca.JR

summary(pca.JR)

## This a PCA biplot, a standard visualization tool in multivariate data analysis
## It allows information on both examples and variables of a data matrix to be displayed simultaneously

## This is a nice visualization tool; if you are interested in it, have a look at:

# http://forrest.psych.unc.edu/research/vista-frames/help/lecturenotes/lecture13/biplot.html

biplot(pca.JR)

## Rough interpretation: indeed all of the ratings are highly correlated, except CONT, which 
## is orthogonal to the rest; proximity of the judges in the 2D representation matches 
## proximity in the original space

## these are the returned singular values of the data matrix
## (the square roots of the eigenvalues of the covariance/correlation matrix)
pca.JR$sdev

eigenval <- pca.JR$sdev^2
xpercent <- eigenval[1]/sum(eigenval)*100   # proportion of variance explained by the first PC
ypercent <- eigenval[2]/sum(eigenval)*100   # proportion of variance explained by the second PC

## Do a PCA plot using the first 2 PCs

plot (pca.JR$x[,1], pca.JR$x[,2], main=paste(paste("PCA -", format(xpercent+ypercent, digits=3)), "%"),
      xlab=paste("1st PC (", format(xpercent, digits=2), "%)"),
      ylab=paste("2nd PC (", format(ypercent, digits=2), "%)"))

text(pca.JR$x[,1], pca.JR$x[,2], rownames(USJudgeRatings), pos= 3)

## We illustrate now a kernelized algorithm: kernel PCA

## first we create a plotting function 

plotting <-function (kernelfu, kerneln)
{
  xpercent <- eig(kernelfu)[1]/sum(eig(kernelfu))*100
  ypercent <- eig(kernelfu)[2]/sum(eig(kernelfu))*100
  
  plot(rotated(kernelfu), 
       main=paste(paste("Kernel PCA (", kerneln, ")", format(xpercent+ypercent,digits=3)), "%"),
       xlab=paste("1st PC -", format(xpercent,digits=3), "%"),
       ylab=paste("2nd PC -", format(ypercent,digits=3), "%"))
  
  text(rotated(kernelfu)[,1], rotated(kernelfu)[,2], rownames(USJudgeRatings), pos= 3)
}

## 1. -------------------------------Linear Kernel---------------------

kpv <- kpca(~., data=USJudgeRatings, kernel="vanilladot", kpar=list(), features=2)
plotting (kpv, "linear")

## 2. ------------------------------Polynomial Kernel (degree 3)-----------------

kpp <- kpca(~., data=USJudgeRatings, kernel="polydot", kpar=list(degree=3,offset=1), features=2)
plotting(kpp,"cubic")

## 3. -------------------------------RBF Kernel-----------------------

kpc1 <- kpca(~., data=USJudgeRatings, kernel="rbfdot", kpar=list(sigma=0.6), features=2)
plotting(kpc1,"RBF - sigma 0.6")

## The effect of sigma is a large one ...
kpc2 <- kpca(~., data=USJudgeRatings, kernel="rbfdot", kpar=list(sigma=0.01), features=2)
plotting(kpc2,"RBF - sigma 0.01")

## It is a pity we do not have legal knowledge about these judges; we could then "judge" the results better
## In particular, some clusters of judges emerge that could be identified (by k-means, for example)
## Also, some trends emerge, as given by the new PCs, in which some judges are at opposite extremes
## This is a drawback of visualization methods, which is aggravated when we go non-linear
## because we now have a tunable parameter


# Now let's do some nice kernel clustering, using spectral clustering, which works by embedding the data into the subspace of the l largest eigenvectors of a normalized kernel matrix and then performing k-means on the embedded points

# Let's exemplify it with the very famous spirals problem:

data(spirals)
summary(spirals)
plot(spirals)

spirals.specc <- specc(spirals, centers=2)
plot(spirals, pch=(23 - 2*spirals.specc))

# The problem of determining the right number of clusters is still open ...

spirals.specc <- specc(spirals, centers=5)
plot(spirals, pch=(23 - 2*spirals.specc))

# Now let's apply the method to our data on US judges; we use the same sigma as for kpc2

USJudges.specc <- specc(as.matrix(USJudgeRatings), centers=2, kernel="rbfdot",kpar=list(sigma=0.01))

plotting.specc <-function (kernelfu, kerneln, colorets)
{
  xpercent <- eig(kernelfu)[1]/sum(eig(kernelfu))*100
  ypercent <- eig(kernelfu)[2]/sum(eig(kernelfu))*100
  
  plot(rotated(kernelfu), 
       main=paste(paste("Kernel PCA (", kerneln, ")", format(xpercent+ypercent,digits=3)), "%"),
       xlab=paste("1st PC -", format(xpercent,digits=3), "%"),
       ylab=paste("2nd PC -", format(ypercent,digits=3), "%"))
  
  text(rotated(kernelfu)[,1], rotated(kernelfu)[,2], rownames(USJudgeRatings), pos= 3, col=colorets)
}

plotting.specc (kpc2,"RBF - sigma 0.01 + SPECC", USJudges.specc@.Data)

USJudges.specc <- specc(as.matrix(USJudgeRatings), centers=4, kernel="rbfdot",kpar=list(sigma=0.01))

plotting.specc (kpc2,"RBF - sigma 0.01 + SPECC", USJudges.specc@.Data)
