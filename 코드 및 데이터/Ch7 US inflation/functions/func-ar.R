runAR=function(Y,horizon,type="fixed"){
  
  Y2=cbind(Y[,1])
  
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
  
  
  if(type=="fixed"){
    model=lm(y~X)
    coef=coef(model)
  }
  
  if(type=="bic"){
    bb=Inf
    for(i in seq(1,ncol(X),1)){
      m=lm(y~X[,1:i])
      crit=BIC(m)
      if(crit<bb){
        bb=crit
        model=m
        ar.coef=coef(model)
      }
    }
    coef=rep(0,ncol(X)+1)
    coef[1:length(ar.coef)]=ar.coef
  }
  pred=c(1,X.out)%*%coef
  
  return(list("model"=model,"pred"=pred,"coef"=coef))
}


ar.rolling.window=function(Y,npred,horizon=1,type="fixed"){
  
  save.coef=matrix(NA,npred-horizon+1,5)   # 수정
  save.pred=matrix(NA,npred-horizon+1,1)   # 수정
  for(i in npred:horizon){                 # 수정
    Y.window=Y[(1+npred-i):(nrow(Y)-i),]
    fact=runAR(Y.window,horizon)
    save.coef[(1+npred-i),]=fact$coef
    save.pred[(1+npred-i),]=fact$pred
    cat("iteration",(1+npred-i),"\n")
  }
  
  # 주어진 windonw가 time T까지라고 하면 T까지의 정보를 이용하여 T+h를 예측. 따라서 각 forecast horizon h마다 생성하는 예측치의 수가 달라짐. 각 h에 대해 npred-horizon+1개 생성. 예를 들어 h=1이면 npred만큼 1-step ahead 예측치를 만들고, h=12라면 npred-12+1개의 12-step ahead 예측치를 만들어냄. 
  
  real=Y[,1]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-npred+horizon-1),save.pred),col="red")   # 수정
  
  rmse=sqrt(mean((tail(real,npred-horizon+1)-save.pred)^2))   # 수정
  mae=mean(abs(tail(real,npred-horizon+1)-save.pred))         # 수정
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred,"coef"=save.coef,"errors"=errors))
}





