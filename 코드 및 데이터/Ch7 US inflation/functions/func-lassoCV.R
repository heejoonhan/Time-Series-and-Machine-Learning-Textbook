runlasso=function(Y,horizon,alpha=1,type="lasso"){
  comp=princomp(scale(Y,scale=FALSE))
  Y2=cbind(Y,comp$scores[,1:4])
  
  X=embed(as.matrix(Y2),4)
  Xin=X[-c((nrow(X)-horizon+1):nrow(X)),]
  Xout=X[nrow(X),]

  y=tail(Y2[,1],nrow(Xin))
  X = Xin 
  X.out = Xout
  ##
  set.seed(4)
  model = cv.glmnet(X,y,family='gaussian',alpha=alpha)  

  if(type=="adalasso"){
    penalty=(abs(coef(model, model$lambda.min)[-1])+1/sqrt(length(y)))^(-1)    
    set.seed(4)
    model=cv.glmnet(X,y,penalty.factor = penalty,alpha=alpha)
  }

  pred=coef(model, model$lambda.min)[1]+X.out%*%(coef(model, model$lambda.min)[-1])
 
  return(list("model"=model,"pred"=pred))
}


lasso.rolling.window=function(Y,npred,horizon=1,alpha=1,type="lasso"){
  
  save.coef=matrix(NA,npred-horizon+1,21+ncol(Y[,-1])*4)
  save.pred=matrix(NA,npred-horizon+1,1)
  for(i in npred:horizon){
    Y.window=Y[(1+npred-i):(nrow(Y)-i),]
    lasso=runlasso(Y.window,horizon,alpha,type)
    #save.coef[(1+npred-i),]=lasso$model$coef
    save.pred[(1+npred-i),]=lasso$pred
    cat("iteration",(1+npred-i),"\n")
  }
  
  real=Y[,1]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-npred+horizon-1),save.pred),col="red")
  
  rmse=sqrt(mean((tail(real,npred-horizon+1)-save.pred)^2))
  mae=mean(abs(tail(real,npred-horizon+1)-save.pred))
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred,"coef"=save.coef,"errors"=errors))
}

