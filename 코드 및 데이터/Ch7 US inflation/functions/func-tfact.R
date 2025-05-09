
run.tfact=function(Y,horizon){
  
  y=Y[,1]
  X=Y[,-1]
  mat=cbind(embed(y,5),tail(X,nrow(X)-4))
  
  pretest=baggit(mat,pre.testing="individual",fixed.controls = 1:4)[-c(1:5)]
  pretest[pretest!=0]=1
  aux=rep(0,ncol(Y))
  aux[1]=1
  aux[-1]=pretest
  selected=which(aux==1)
  
  Y2=Y[,selected]
  
  fmodel=runfact(Y2,horizon=horizon)
  
  coef=fmodel$coef
  pred=fmodel$pred
  
  return(list("coef"=coef,"pred"=pred))
}


tfact.rolling.window=function(Y,npred,horizon=1){
  
  save.coef=matrix(NA,npred-horizon+1,21)
  save.pred=matrix(NA,npred-horizon+1,1)
  for(i in npred:horizon){
    Y.window=Y[(1+npred-i):(nrow(Y)-i),]
    fact=run.tfact(Y.window,horizon)
    #save.coef[(1+npred-i),]=fact$coef
    save.pred[(1+npred-i),]=fact$pred
    cat("iteration",(1+npred-i),"\n")
  }
  
  real=Y[,1]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-npred),save.pred),col="red")
  
  rmse=sqrt(mean((tail(real,npred-horizon+1)-save.pred)^2))
  mae=mean(abs(tail(real,npred-horizon+1)-save.pred))
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred,"errors"=errors))
}

