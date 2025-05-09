run_multi_lstm=function(Y,horizon){
  
  comp=princomp(scale(Y,scale=FALSE))
  Y2 = cbind(Y, comp$scores[,1:4]) %>% as.data.frame()
  Y3 = lapply(Y2, normalize) %>% as.data.frame() %>% as.matrix()
  
  # 아래 부분은 old version으로 X.out을 정의할 때 horizon>1인 경우 오류가 있음 
  #aux=embed(Y3,4+horizon)
  #y=aux[,indice]
  #X=aux[,-c(1:(ncol(Y3)*horizon))]  
  
  #if(horizon==1){
  #  X.out=tail(aux,1)[1:ncol(X)]  
  #}else{
  #  X.out=aux[,-c(1:(ncol(Y3)*(horizon-1)))]
  #  X.out=tail(X.out,1)[1:ncol(X)]
  #}
  
  X=embed(as.matrix(Y3),4)
  
  Xin=X[-c((nrow(X)-horizon+1):nrow(X)),]
  Xout=X[nrow(X),]
  #Xout=t(as.vector(Xout))  # 이게 필요하나?
  
  y=tail(Y3[,1],nrow(Xin))
  X = Xin 
  X.out = Xout
  ##
  
  
  
  ###
  X2 <- X %>% replace(!0, 0) 
  
  for(i in 0:(ncol(Y3)-1)){
    X2[,(4*i+4)] <- X[,(i+1)]
    X2[,(4*i+3)] <- X[,(i+ncol(Y3)+1)]
    X2[,(4*i+2)] <- X[,(i+2*ncol(Y3)+1)]
    X2[,(4*i+1)] <- X[,(i+3*ncol(Y3)+1)]
  }
  
  X.out2 <- X.out %>% replace(!0, 0)
  
  for(i in 0:(ncol(Y3)-1)){
    X.out2[(4*i+4)] <- X.out[(i+1)]
    X.out2[(4*i+3)] <- X.out[(i+ncol(Y3)+1)]
    X.out2[(4*i+2)] <- X.out[(i+2*ncol(Y3)+1)]
    X.out2[(4*i+1)] <- X.out[(i+3*ncol(Y3)+1)]
  }
  
  ###  
  X.arr = array(
    data = as.numeric(unlist(X2)),
    dim = c(nrow(X), 4, ncol(Y3)))
  
  X.out.arr = array(
    data = as.numeric(unlist(X.out2)),
    dim = c(1, 4, ncol(Y3)))
  
  # =============================================================
  set.seed(42)         # 신경망의 Initial weight를 설정할 때, Dropout과 같은 방법을 사용할 때 등에서 영향 
  set_random_seed(42)  #tensorflow의 경우 tensorflow::set_random_seed()  함수를 사용해서 시드 설정
  
  
  
  # Hyper-Parameters Adjustment
  
  batch_size = 50  #  한 번에 입력하는 데이터 크기 
  feature = ncol(Y3)  # 설명변수 수 
  epochs = 100  # 학습 횟수, 100
  
  model = keras_model_sequential()
  
  # 1-layer model 실행
  
  model %>% layer_lstm(units = 32, 
                       input_shape = c(4, feature),
                       stateful = FALSE) %>%
    layer_dense(units = 1) 
  
  
  # 2-layer model with drop out (rate = 0.3)   ??   κ      s = 32,
  #                      input_shape = c(4, feature),
  #                      stateful = FALSE,
  #                      return_sequences = TRUE) %>% 
  #   layer_dropout(rate = 0.3) %>% 
  #   layer_lstm(units = 16) %>% 
  #   layer_dropout(rate = 0.3) %>% 
  #   layer_dense(units = 1)
  
  model %>% compile(loss = 'mse', optimizer = 'adam')
  
  model %>% summary()
  
  history = model %>% fit(X.arr, y, epochs = epochs,
                          batch_size = batch_size, shuffle = FALSE, verbose = 2)
  
  # =============================================================
  
  pred = model(X.out.arr) %>% denormalize(min(Y2[,1]),max(Y2[,1])) # De-normalization
  
  return(list("model"=model,"pred"=pred))
}



mul.lstm.rolling.window=function(Y,npred,horizon=1){
  
  save.pred=matrix(NA,npred-horizon+1,1)
  for(i in npred:horizon){
    Y.window=Y[(1+npred-i):(nrow(Y)-i),] %>% as.data.frame()
    lstm=run_multi_lstm(Y.window,horizon)
    save.pred[(1+npred-i),]=as.numeric(lstm$pred) # Note as.numeric()
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