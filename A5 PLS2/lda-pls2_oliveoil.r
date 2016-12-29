# PLS2 on Olive Oil Sensory Analysis
par(mfrow=c(1,1))

library(pls)
library(calibrate)
library(nnet)
library(MASS)

data(oliveoil)

class(oliveoil)

dim(oliveoil)

names(oliveoil)

dim(oliveoil$chemical)
dim(oliveoil$sensory)

head(oliveoil)

# PREDICTION OF ORIGIN FROM CHEMICAL DATA

# forming of the predictorand response matrices

X <- oliveoil$chemical
y <- c(rep("greek",5),rep("italian",5),rep("spanish",6))
Y <- class.ind(y)

print(X)
print(Y)


n <- nrow(X)
p <- ncol(X)
q <- ncol(Y)


# standardization of data 

Xs <- as.matrix(scale(X))
#Ys <- as.matrix(scale(Y))
Ys <- Y

# PLS2

p2 <- plsr(Ys ~ Xs, validation = "LOO")

summary(p2)

# selecting the number of components

plot(R2(p2), legendpos = "bottomright")

R2.cv <- R2(p2)$val[1,,]
R2.cv

plot(apply(R2.cv,2,mean),type="l",xaxt="n")

axis(1,at=1:ncol(R2.cv),lab=colnames(R2.cv),tick=FALSE)

nd <- which.max(apply(R2.cv,2,mean))-1



# looking for the approximated cv R2

for (i in 1:p) {lmY <- lm(Ys~p2$scores[,1:i])
                PRESS  <- apply((lmY$residuals/(1-ls.diag(lmY)$hat))^2,2,sum)
                RMPRESS <- sqrt(PRESS/n)
                print(paste("N. of components ",i))
                print("RMSEP :")
                print(RMPRESS,digits=4)
                R2cv   <- 1 - PRESS/(sd(Ys)^2*(n-1))
                print("R2cv : ")
                print(R2cv,digits=4) }


# scores plot
plot(p2, plottype = "scores", comps = 1:2, type="n", main="X Scores")
text(p2$scores, labels=rownames(p2$scores), col=as.numeric(as.factor(y)))
axis(side=1, pos= 0, labels = F, col="gray")
axis(side=3, pos= 0, labels = F, col="gray")
axis(side=2, pos= 0, labels = F, col="gray")
axis(side=4, pos= 0, labels = F, col="gray")
legend("topright",c(levels(as.factor(y))),col=c(1:3),lty=1)


# loading plot
plot(p2, "loadings", comps = 1:nd, legendpos = "topleft", labels = rownames(p2$loadings))
abline(h = 0)



# coefficients plot
plot(p2, plottype="coef", ncomp = nd, legendpos = "topleft", labels = rownames(p2$coefficients))


p2$coefficients[,,nd]


# correlation plot

# plot of correlations

corXp2 <- cor(Xs,p2$scores)
corYp2 <- cor(Ys,p2$scores)
corXYp2 <- rbind(corXp2,corYp2)
plot(corXYp2,ylim=c(-1,1),xlim=c(-1,1),asp=1,type="n",main="Correlations with components")
text(corXYp2,labels=rownames(corXYp2),col=c(rep(1,p),rep(2,q)),adj=1.1,cex=0.85)
arrows(rep(0,(p+1)),rep(0,(p+1)),corXYp2[,1],corXYp2[,2],col=c(rep(1,p),rep(2,q)),length=0.07)
axis(side=1,pos=0,labels=F,col="gray")
axis(side=2,pos=0,labels=F,col="gray")
axis(side=3,pos=0,labels=F,col="gray")
axis(side=4,pos=0,labels=F,col="gray")
circle()



# prediction through discriminant analysis
da <- lda(p2$scores[,1:nd],y,CV=TRUE)

table(y,da$class)
pred.chem = data.frame(y,da$class,da$posterior)
print(pred.chem,digits=4)

chem <- p2$scores[,1:nd]




# PREDICTION OF ORIGIN FROM SENSORY DATA


# forming of the predictorand response matrices

X <- oliveoil$sensory


print(X)
print(Y)


n <- nrow(X)
p <- ncol(X)
q <- ncol(Y)


# standardization of data 

Xs <- as.matrix(scale(X))
#Ys <- as.matrix(scale(Y))
Ys <- Y

# PLS2

p2 <- plsr(Ys ~ Xs, validation = "LOO")

summary(p2)

# selecting the number of components

plot(R2(p2), legendpos = "bottomright")

R2cv <- R2(p2)$val[1,,]

plot(apply(R2cv,2,mean),type="l",xaxt="n")
axis(1,at=1:ncol(R2cv),lab=colnames(R2cv),tick=FALSE)

nd <- which.max(apply(R2cv,2,mean))-1



# looking for the approximated cv R2

for (i in 1:p) {lmY <- lm(Ys~p2$scores[,1:i])
                PRESS  <- apply((lmY$residuals/(1-ls.diag(lmY)$hat))^2,2,sum)
                RMPRESS <- sqrt(PRESS/n)
                print(paste("N. of components ",i))
                print("RMSEP :")
                print(RMPRESS,digits=4)
                R2cv   <- 1 - PRESS/(sd(Ys)^2*(n-1))
                print("R2cv : ")
                print(R2cv,digits=4) }

# prediction plot
plot(p2, ncomp = nd, asp = 1, line = TRUE, col=c(rep(1,5),rep(2,5),rep(3,6)))


# scores plot
plot(p2, plottype = "scores", comps = 1:2, type="n", main="X Scores")
text(p2$scores, labels=rownames(p2$scores), col=as.numeric(as.factor(y)))
axis(side=1, pos= 0, labels = F, col="gray")
axis(side=3, pos= 0, labels = F, col="gray")
axis(side=2, pos= 0, labels = F, col="gray")
axis(side=4, pos= 0, labels = F, col="gray")
legend("bottomleft",c(levels(as.factor(y))),col=c(1:3),lty=1)


# loading plot
plot(p2, "loadings", comps = 1:nd, legendpos = "topleft", labels = rownames(p2$loadings))
abline(h = 0)


# coefficients plot
plot(p2, plottype="coef", ncomp = nd, legendpos = "topleft", labels = rownames(p2$coefficients))

p2$coefficients[,,nd]


# correlation plot

# plot of correlations

corXp2 <- cor(Xs,p2$scores)
corYp2 <- cor(Ys,p2$scores)
corXYp2 <- rbind(corXp2,corYp2)
plot(corXYp2,ylim=c(-1,1),xlim=c(-1,1),asp=1,type="n",main="Correlations with components")
text(corXYp2,labels=rownames(corXYp2),col=c(rep(1,p),rep(2,q)),adj=1.1,cex=0.85)
arrows(rep(0,(p+1)),rep(0,(p+1)),corXYp2[,1],corXYp2[,2],col=c(rep(1,p),rep(2,q)),length=0.07)
axis(side=1,pos=0,labels=F,col="gray")
axis(side=2,pos=0,labels=F,col="gray")
axis(side=3,pos=0,labels=F,col="gray")
axis(side=4,pos=0,labels=F,col="gray")
circle()




# prediction from sensory data by lda of pls_components


da <- lda(p2$scores[,1:nd],y,CV=TRUE)
table(y,da$class)
pred.sens = data.frame(y,da$class,da$posterior)
print(pred.sens,digits=4)

sens <- p2$scores[,1:nd]


# lda PREDICTION USING CHEMICAL AND SENSORY COMPONENTS


da <- lda(cbind(chem,sens),y,CV=TRUE)
table(y,da$class)
pred.tot = data.frame(y,da$class,da$posterior)
print(pred.tot,digits=4)
