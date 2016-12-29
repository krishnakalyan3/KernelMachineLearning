interbatt = function(X,Y){
  Vxy = var(X,Y)
  rank = qr(Vxy)$rank
  print(paste("Rank of CoVar Matrix ",rank))
  aib = eigen(t(Vxy)%*%Vxy)
  A = Vxy %*% aib$vectors %*% diag(aib$values^(-0.5))
  TIB = as.matrix(X) %*% A
  rl = list(TIB, A)
  return(rl)
}

eval_func = function(y, yhat, cm_show = FALSE){
  metrics = c()
  cm = table(y,yhat)
  if(cm_show == TRUE){
    print(cm)
  }
  total = sum(cm)
  no_diag = cm[row(cm) != (col(cm))]
  acc = sum(diag(cm))/total
  error = sum(no_diag)/total
  metrics = c(acc,error)
  return(metrics)
}

plot_img = function(train_data,i,show_target=FALSE){
  CUSTOM_COLORS = colorRampPalette(colors = c("black", "white"))
  if(show_target==TRUE){
    print(train_data[i,1])
  }
  train_data = train_data[,-1]
  z = unname(unlist((train_data[i,])))
  k = matrix(z,nrow = 16,ncol = 16)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(t(k)), col = CUSTOM_COLORS(256))
}