library(plyr)

fix_levels = function(train, test){
  for(i in names(test)){
    if (length(levels(test[,i])) < length(levels(train[,i]))){
      levels(test[,i]) = levels(train[,i])
    }else{
      levels(train[,i]) = levels(test[,i])
    }
  }
  return(list(train,test))
} 


kfold_svr = function(data, k = 5, model=ksvm, formuale){
  svr_model = model(formuale, kernel='rbfdot', data=data, type='nu-svr', cross = 5, scaled= T)
  data$fold = sample(1:k, nrow(data), replace = TRUE)
  list <- 1:k
  progress.bar <- create_progress_bar("text")
  progress.bar$init(k)
  mae_all = matrix(k)
  for (i in 1:k){
    trainingset = subset(data, fold %in% list[-i])
    testset = subset(data, fold %in% c(i))
    model1 = model(formuale, data = trainingset, kernel='rbfdot', type='nu-svr', cross = 5, scaled= TRUE)
    yhat = predict(model1, testset)
    mae = mean(abs(yhat - exp(testset$loss)))
    mae_all[i] = mae
    progress.bar$step()
  }
  mae_avg = mean(mae_all)
  print(paste("Avg MAE ", mae_avg))

  return(list(svr_model, mae_avg))
}

