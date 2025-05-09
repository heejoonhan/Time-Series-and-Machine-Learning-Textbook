runfact=function(Y,horizon){
  comp=princomp(scale(Y,scale=FALSE))
  Y2=cbind(Y,comp$scores[,1:4])
  
  # ?Ʒ? ?κ?�� old version��?? X.out�� ��???? ?? horizon>1?? ???? ?��??? ??�� 
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
  #Xout=t(as.vector(Xout))  # ?̰? ?ʿ??ϳ??
  
  y=tail(Y2[,1],nrow(Xin))
  X = Xin 
  X.out = Xout
  ##
  
  bb=Inf
  for(i in seq(5,20,5)){
    m=lm(y~X[,1:i])
    crit=BIC(m)
    if(crit<bb){
      bb=crit
      model=m
      f.coef=coef(model)
    }
  }
  coef=rep(0,ncol(X)+1)
  coef[1:length(f.coef)]=f.coef
  
  pred=c(1,X.out)%*%coef
  
  return(list("model"=model,"pred"=pred))
}



fact.rollin=coefdow=function(Y,npred,horizon=1){
  
  save.coef=matrix(NA,npred-horizon+1,21)
  save.pred=matrix(NA,npred-horizon+1,1)
  for(i in npred:horizon){
    Y.window=Y[(1+npred-i):(nrow(Y)-i),]
    fact=runfact(Y.window,horizon)
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