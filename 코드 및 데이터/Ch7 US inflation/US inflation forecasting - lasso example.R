######################################
### 2. Forecasting part: LASSO example
######################################

setwd("D:/0교재 집필/R code/Ch7 US inflation")  
dir()

rm(list=ls())
FREDMD <- read_excel("FREDMDdata.xlsx",sheet=1)
Y <- as.matrix(FREDMD)

# number of forecasts 
npred = 132 
# window size = nrow(Y) - npred 

## LASSO, cross-validation 사용 ##
library(glmnet)
source("functions/func-lassoCV.R")

alpha=1

## 1개월~3개월 앞 예측
lasso1=lasso.rolling.window(Y,npred,1,alpha,type="lasso")
laslasso2=lasso.rolling.window(Y,npred,2,alpha,type="lasso")
lasso3=lasso.rolling.window(Y,npred,3,alpha,type="lasso")


runlasso=function(Y,horizon,alpha=1,type="lasso"){
  # 주성분 결합
  comp=princomp(scale(Y,scale=FALSE))
  Y2=cbind(Y,comp$scores[,1:4])
  
  X=embed(as.matrix(Y2),4)
  Xin=X[-c((nrow(X)-horizon+1):nrow(X)),]
  X.out=X[nrow(X),]
  y=tail(Y2[,1],nrow(Xin))
  
  # 라쏘 또는 적응형 라쏘
  set.seed(4)
  model = cv.glmnet(Xin,y,family='gaussian',alpha=alpha)  
  
  if(type=="adalasso"){
    penalty=(abs(coef(model,model$lambda.min)[-1])
             +1/sqrt(length(y)))^(-1)    
    set.seed(4)
    model=cv.glmnet(Xin,y,penalty.factor=penalty,alpha=alpha)
  }
  
  # 예측치 계산
  pred=as.numeric(c(1, X.out)%*%coef(model,model$lambda.min))
  
  return(list("model"=model,"pred"=pred,
              "coef"=coef(model,model$lambda.min)))
}


lasso.rolling.window=function(Y,npred,horizon=1,alpha=1,type="lasso"){
  
  save.coef=matrix(NA,npred-horizon+1,21+ncol(Y[,-1])*4)
  save.pred=matrix(NA,npred-horizon+1,1)
  
  # rolling window forecasting
  for(i in npred:horizon){
    Y.window=Y[(1+npred-i):(nrow(Y)-i),]
    lasso=runlasso(Y.window,horizon,alpha,type)
    save.coef[(1+npred-i),]=as.numeric(lasso$coef)
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







