# Fixed PCA
library(FactoMineR)
library(caTools)
library(doMC)
library(caret)
registerDoMC(4)


set.seed(1337)
setwd('/Users/krishna/MIRI/KML')

train = read.csv('data/train.csv')
train$loss = log(train$loss)
test =  read.csv('data/test.csv')

remove_col=c('id','loss')
datatrain = train[,-which(names(train) %in% remove_col)]
datatest =  test[,-which(names(test) %in% remove_col)]
numeric_col = sapply(datatrain, is.numeric) 

datatrain_num = datatrain[,numeric_col]
datatest_num  = datatest[,numeric_col]

# Scale
datatrain_scale = scale(datatrain_num, center = T, scale = F)
datatest_scale = scale(datatest_num, center = colMeans(datatrain_num), scale = F)
par(mfrow = c(1, 2))

# PCA
plot(1:10)
pca = PCA(datatrain_scale , ncp = 13)
# Test Projections
test_proj = predict(pca,datatest_num)

eigenvalues = pca$eig
barplot(eigenvalues[, 2], names.arg=1:nrow(eigenvalues), 
        main = "Variances",
        xlab = "Principal Components",
        ylab = "Percentage of variances",
        col ="steelblue")

lines(x = 1:nrow(eigenvalues), eigenvalues[, 2], 
      type="b", pch=19, col = "red")

train_comp = data.frame(pca$ind$coord)
test_proj = predict(pca,datatest)
test_comp = data.frame(test_proj$coord)

# Save
saveRDS(pca, 'data/pca/pca_res.rds')
write.csv(train_comp, "data/pca/pca_train_comp.csv", row.names=FALSE)
write.csv(test_comp, "data/pca/pca_test_comp.csv", row.names=FALSE)
