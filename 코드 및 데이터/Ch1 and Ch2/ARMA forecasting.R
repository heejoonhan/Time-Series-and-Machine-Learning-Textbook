
## Forecasting of ARMA models 
library(forecast)

# AR(2)
set.seed(40)
sim.ar=arima.sim(list(order=c(2,0,0),ar=c(0.6,0.2)),n=500)

# true DGP: AR(2)
y = sim.ar

## Rolling window forecasting
w_size = 250
nf = length(y) - w_size # number of forecasts 

AR2_pred=matrix(0,nf,1)
for(i in 1:nf){ 
  y_window = y[i:(w_size-1+i)] 
  fit1=arima(y_window,order=c(2,0,0))
  AR2_pred[i] = predict(fit1)$pred
}

MA2_pred=matrix(0,nf,1)
for(i in 1:nf){ 
  y_window = y[i:(w_size-1+i)] 
  fit2=arima(y_window,order=c(0,0,2))
  MA2_pred[i] = predict(fit2)$pred
}


# Comparison of forecast loss
real=tail(y,nf) # actual value

mse_AR2=mean((real-AR2_pred)^2)
mse_MA2=mean((real-MA2_pred)^2)

mae_AR2=mean(abs(real-AR2_pred))
mae_MA2=mean(abs(real-MA2_pred))

cbind(mse_AR2, mse_MA2)
cbind(mae_AR2, mae_MA2)

## DMW test
source("dmwtest.r")
dmwtest(real,AR2_pred,MA2_pred)

## GW test
source("gwtest.r")
gw.test(AR2_pred,MA2_pred,real,tau=1,T=nf,method="NeweyWest")

