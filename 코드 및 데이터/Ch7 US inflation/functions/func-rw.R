
rw.rolling.window=function(Y,npred,horizon=1){
  
  save.pred=matrix(NA,npred-horizon+1,1)
  for(i in npred:horizon){
    Y.window=Y[(1+npred-i):(nrow(Y)-i),]
    pred=tail(Y.window[,1],1)

    save.pred[(1+npred-i),]=pred
    cat("iteration",(1+npred-i),"\n")
    
  }
  
  real=Y[,1]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-npred+horizon-1),save.pred),col="red")
  
  rmse=sqrt(mean((tail(real,npred-horizon+1)-save.pred)^2))
  mae=mean(abs(tail(real,npred-horizon+1)-save.pred))
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred, "errors"=errors))
  
}
