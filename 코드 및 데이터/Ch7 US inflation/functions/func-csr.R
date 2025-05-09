
runcsr=function(Y,horizon){
  comp=princomp(scale(Y,scale=FALSE))
  Y2=cbind(Y,comp$scores[,1:4])
  
  # 아래 부분은 old version으로 X.out을 정의할 때 horizon>1인 경우 오류가 있음 
  #aux=embed(Y2,4+horizon)
  #y=aux[,1]
  #X=aux[,-c(1:(ncol(Y2)*horizon))]  
  
  #if(horizon==1){
  #  X.out=tail(aux,1)[1:ncol(X)]  
  #}else{
  #  X.out=aux[,-c(1:(ncol(Y2)*(horizon-1)))]
  #  X.out=tail(X.out,1)[1:ncol(X)]
  #}
  
  
  X=embed(as.matrix(Y2),4)
  
  Xin=X[-c((nrow(X)-horizon+1):nrow(X)),]
  Xout=X[nrow(X),]
  #Xout=t(as.vector(Xout))  # 이게 필요하나?
  
  y=tail(Y2[,1],nrow(Xin))
  X = Xin 
  X.out = Xout
  ##

  
  f.seq=seq(1,ncol(X),ncol(Y2))
  
  model=HDeconometrics::csr(x=X,y,fixed.controls = f.seq)
  pred=predict(model,X.out)
  
  return(list("model"=model,"pred"=pred))
}


csr.rolling.window=function(Y,npred,horizon=1){
  
  save.pred=matrix(NA,npred-horizon+1,1)
  for(i in npred:horizon){
    Y.window=Y[(1+npred-i):(nrow(Y)-i),]
    cs=runcsr(Y.window,horizon)
    save.pred[(1+npred-i),]=cs$pred
    cat("iteration",(1+npred-i),"\n")
  }
  
  real=Y[,1]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-npred+horizon-1),save.pred),col="red")
  
  rmse=sqrt(mean((tail(real,npred-horizon+1)-save.pred)^2))
  mae=mean(abs(tail(real,npred-horizon+1)-save.pred))
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred,"errors"=errors))
}
