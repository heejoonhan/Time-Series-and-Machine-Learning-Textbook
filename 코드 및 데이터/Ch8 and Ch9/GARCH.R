## GARCH, GJR-GARCH, GARCH with t-distribution model estimation ##

getwd()
setwd("")
dir() 

#install.packages("readxl")
#install.packages("fBasics")

library("readxl")
library("fBasics")

data <- read_excel("S_Pdata.xls")
head(data)
tail(data)

sp=matrix(data$SPX_r)

plot(sp, type = 'l', xlab="")

# descriptive statistics for the return series
kurtosis(sp)


# GARCH(1,1) #
install.packages('fGarch')
library(fGarch)

fit1=garchFit(~garch(1,1),include.mean=T,data=sp,trace=F,cond.dist="QMLE") 
fit1
summary(fit1)
ts.plot(fit1@h.t)


# GJR-GARCH(1,1) #
install.packages('rugarch')
library(rugarch)
?ugarchspec

gjr.garch = ugarchspec(variance.model=list(model="gjrGARCH", garchOrder=c(1,1)),
                      mean.model=list(armaOrder=c(0,0),include.mean=T))
gjr.garch

fit2 = ugarchfit(gjr.garch,sp)
fit2


# GARCH(1,1) with t-distribution #
garch.t = ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,1)),
                       mean.model=list(armaOrder=c(0,0),include.mean=T),
                     distribution.model = "std")

fit3 = ugarchfit(garch.t,sp)
fit3


