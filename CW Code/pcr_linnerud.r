# loading of the packages
#install.packages('FactoMineR')
library(pls)
library(calibrate)
library(FactoMineR)

# loading the data

#setwd("E:/txt/docent/DMKM/KBLMM/2016-2017/Session 1 pcr/")
setwd("/Users/krishna/MIRI/MVA/linnerud")

linnerud <- read.table("linnerud.txt",header=T)
head(linnerud)

# forming of the predictorand response matrices

X <- linnerud[,1:3]
Y <- linnerud[,4:6]

print(X)
print(Y)


n <- nrow(X)
p <- ncol(X)
q <- ncol(Y)


# standardization of data 

Xs <- as.matrix(scale(X))
Ys <- as.matrix(scale(Y))

# MULTIVATIATE REGRESSION

mreg <- lm(Ys ~ Xs-1)

print(mreg)
cor(X,Y)

summary(manova(mreg))

summary(mreg)

# R2 by LOO

PRESS  <- colSums((mreg$residuals/(1-ls.diag(mreg)$hat))^2)
R2cv   <- 1 - PRESS/(diag(var(Ys))*(n-1))

R2cv


# lets do first 2 PCAs. First PCA of X

pcX <- PCA(linnerud, quanti.sup=4:6)

attributes(pcX)

# table of eigenvalues

pcX$eig
windows()
plot(pcX$eig$eigenvalue,type="l", main="Screeplot")   #screeplot

# correlations variable - principal component

print(pcX$var$cor)

# compare with the coordinates of variable in a normalized ACP

print(pcX$var$coord)

print("Enter the significant number of dimensions")
nd <- scan (what=integer(), nmax=1, quiet=T)

if (nd<2) nd = 2

# we store the loadings and scores in U and psi
P = pcX$svd$V[,1:nd]
Psi = pcX$ind$coord[,1:nd]
eigv = pcX$eig$eigenvalue[1:nd]

iden = row.names(X)

etiq = colnames(X)
   
ze = rep(0,length(etiq))

# biplot in Rp
windows()
plot(Psi[,1],Psi[,2],type="n",asp=1,main="Biplot in Rp")
text(Psi[,1],Psi[,2],labels=iden)
abline(h=0,v=0,col="gray")
arrows(ze, ze, P[,1], P[,2], length = 0.07,col="blue")
text(P[,1],P[,2],labels=etiq,col="blue")
circle(1)


# biplot in Rn

cor.VF = pcX$var$cor[,1:nd]
cor.VF

Psis = Psi %*% diag(1/sqrt(eigv))

windows()
plot(cor.VF[,1],cor.VF[,2],ylim=c(-2,3),xlim=c(-2,3),asp=1,type="n",main="Biplot in Rn")
abline(h=0,v=0,col="gray")
arrows(ze, ze, cor.VF[,1], cor.VF[,2], length = 0.07, col="blue")
text(cor.VF[,1],cor.VF[,2],labels=etiq, col="blue")
text(Psis[,1],Psis[,2],pch=20,labels=iden,col="gray")
circle(1)

# postioning of the suplementary variables

cor.VFsup = cor(Y,Psi)
cor.VFsup
etiqsup = colnames(Y)

arrows(ze, ze, cor.VFsup[,1], cor.VFsup[,2], length = 0.07, col="violet",lty=2)
text(cor.VFsup[,1],cor.VFsup[,2],labels=etiqsup, col="violet")

# rotation of axes to find the latent variables

pcrot = varimax(cor.VF)
print(pcrot)

cor.VFrot = pcrot$loadings[1:p,]
windows()
plot(cor.VFrot[,1],cor.VFrot[,2],ylim=c(-1,1),xlim=c(-1,1),asp=1,type="n",main="Correlations with components")
abline(h=0,v=0,col="gray")
arrows(ze, ze, cor.VFrot[,1], cor.VFrot[,2], length = 0.07, col="blue")
text(cor.VFrot[,1],cor.VFrot[,2],labels=etiq, col="blue")
circle(1)


Psirot = Xs %*% solve(cor(Xs)) %*% cor.VFrot

# plot of individuals in the rotated axes in Rp
windows()
plot(Psirot[,1],Psirot[,2],type="n",asp=1,main="Rotated plot of individuals in Rp")
text(Psirot[,1],Psirot[,2],labels=iden)
abline(h=0,v=0,col="gray")


# PCA of block Y   (just for explorative purposes)

pcY <- PCA(linnerud, quanti.sup=1:3)

# table of eigenvalues

pcY$eig
windows()
plot(pcY$eig$eigenvalue,type="l", main="Screeplot")   #screeplot

print("Enter the significant number of dimensions")
nd <- scan (what=integer(), nmax=1, quiet=T)

if (nd<2) nd = 2

# we store the loadings and scores in U and psi
PY = pcY$svd$V[,1:nd]
PsiY = pcY$ind$coord[,1:nd]

idenY = row.names(Y)

etiqY = colnames(Y)
   
ze = rep(0,length(etiqY))

# biplot in Rp
plot(PsiY[,1],PsiY[,2],type="n",asp=1,main="Biplot in Rp")
text(PsiY[,1],PsiY[,2],labels=iden)
abline(h=0,v=0,col="gray")
arrows(ze, ze, PY[,1], PY[,2], length = 0.07,col="blue")
text(PY[,1],PY[,2],labels=etiqY,col="blue")
circle(1)


# correlation plot (biplot in Rn)

cor.VFY = cor(Y,PsiY)
cor.VFY

plot(cor.VFY[,1],cor.VFY[,2],ylim=c(-1,1),xlim=c(-1,1),asp=1,type="n",main="Correlations with components")
abline(h=0,v=0,col="gray")
arrows(ze, ze, cor.VFY[,1], cor.VFY[,2], length = 0.07, col="blue")
text(cor.VFY[,1],cor.VFY[,2],labels=etiq, col="blue")
circle(1)



###########################################################3
# PCR
# now the PCR with all components and leave one out crossvalidation

pc <- pcr(Ys~Xs, validation="LOO")

attributes(pc)

summary(pc)


#compare
pc$scores %*% t(pc$loadings)    # Xs = T * P'
Xs

# compare
pc$loadings[1:p,]         # = P
pcX$svd$V
lm(Xs~pc$scores-1)     # Xs = T * P'    = Biplot in Rp
lm(pc$scores~Xs-1)     # T = Xs * P     = definition of pcr components

#compare
pc$Yloadings[1:q,]
lm(Ys~pc$scores-1)

# correlations of X and Y with pcr components T

corXpc <- cor(Xs,pc$scores)
corXpc
corYpc <- cor(Ys,pc$scores)
corYpc

# communalities of X           
apply(corXpc^2,1,cumsum)    
rowMeans(apply(corXpc^2,1,cumsum))

# redundancies of Y

apply(corYpc^2,1,cumsum)   
rowMeans(apply(corYpc^2,1,cumsum))



# looking for the significant components

plot(R2(pc), legendpos = "topright")

# looking for the cv R2

plot(R2(pc), legendpos = "topright")
R2.cv <- R2(pc)$val[1,,]
R2.cv
plot(apply(R2.cv,2,mean),type="l")

# we take 1 component

nd = 1

# fitted values versus observed

plot(pc, ncomp = 1:nd, asp = 1, line = TRUE)


# plot of loadings

plot(pc, "loadings", comps = 1:nd, legendpos = "topleft", labels = rownames(pc$loadings))
abline(h = 0)

# plot of regression coefficients

plot(pc, plottype = "coef", ncomp = nd, legendpos = "bottomleft", labels = rownames(pc$loadings))


# plot of correlations

corXYpc <- rbind(corXpc,corYpc)
plot(corXYpc[,1],corXYpc[,2],ylim=c(-1,1),xlim=c(-1,1),asp=1,type="n",main="Correlations with components")
text(corXYpc[,1],corXYpc[,2],labels=rownames(corXYpc),col=c(rep(1,p),rep(2,q)),adj=1.1,cex=0.85)
arrows(rep(0,(p+q)),rep(0,(p+q)),corXYpc[,1],corXYpc[,2],col=c(rep(1,p),rep(2,q)),length=0.07)
abline(h=0,v=0,col="gray")
circle(1)



# fit of Y on the first Principal Component

lmY <- lm(Ys~pc$scores[,1:nd]-1)
summary(lmY)

summary(manova(lmY))


b <- lmY$coefficients[1:nd,]
b

p1 <- pc$loadings[,1:nd]

as.matrix(p1)%*%t(as.matrix(b))
pc$coefficients[,,nd]

cor(X,Y)




