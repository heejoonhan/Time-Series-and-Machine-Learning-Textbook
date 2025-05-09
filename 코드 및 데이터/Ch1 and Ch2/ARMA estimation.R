## Estimation of ARMA models 

# 1) simulate ARIMA(p,d,q) models
# ARMA(1,1)
sim.arma=arima.sim(list(order=c(1,0,1),ar=0.5,ma=0.5),n=1000)

# AR(2)
set.seed(40)
sim.ar=arima.sim(list(order=c(2,0,0),ar=c(0.6,0.2)),n=1000)

# true DGP: AR(2)
y = sim.ar
plot(y)


# 2) estimate ARIMA(p,d,q) models
fit1=arima(y,order=c(2,0,0))
fit1

# Ljung-Box Q-test
uhat<-fit1$resid
Box.test(uhat,lag=10,type='Ljung') 


# misspecified case
fit2=arima(y,order=c(0,0,2))
fit2

uhat2<-fit2$resid
Box.test(uhat2,lag=10,type='Ljung') 


# 3) auto.arima ÁÖÀÇ
install.packages("forecast")
library(forecast)

autofit=auto.arima(y, seasonal=FALSE)
autofit



