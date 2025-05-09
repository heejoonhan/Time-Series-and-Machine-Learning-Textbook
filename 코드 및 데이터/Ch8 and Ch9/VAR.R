########################################
#Park and Ratti, 2008, Energy Economics#
#VAR, IRF, and Variance Decomposition###
########################################

getwd()
setwd("D:/0교재 집필/R code/Ch 8and9") 
dir() 

install.packages("vars")
library(vars)

####################
## 0. Loading Data #
####################

data=read.table('United States.txt',header=TRUE,fill=TRUE)       

r=as.matrix(data$r)
op_w=as.matrix(data$op_w)
ip=as.matrix(data$ip)
rsr=as.matrix(data$rsr)
y=cbind(r,op_w,ip,rsr)[-(1),]                                   
colnames(y) <- c("r","op_w","ip","rsr")
head(y)

##################################
## 1. Estimation of VAR(p) Model #
##################################

VARselect(y,lag.max=12,type="const")

VAR3=VAR(y,p=3,type=c("const"),season=NULL,exogen=NULL) 
# summary(VAR3)

# checking residual
serl <- serial.test(VAR3, lags.pt = 12, type = "PT.asymptotic")
serl$serial

################################
# 2. Impulse-Response Analysis #
################################

y.irf95=irf(VAR3,impulse="op_w",response="rsr",n.ahead=24,ortho=TRUE, 
          cumulative=FALSE,boot=TRUE,ci=0.95,runs=200,seed=NULL)
plot(y.irf95)

y.irf68=irf(VAR3,impulse="op_w",response="rsr",n.ahead=24,ortho=TRUE, 
          cumulative=FALSE,boot=TRUE,ci=0.68,runs=200,seed=NULL)
plot(y.irf68)


############################
# 3.Variance Decomposition #
############################

fevd(VAR3,n.ahead=24)$rsr                        

############################
# 4.Granger Causality Test #
############################

# H0: rsr does not Granger cause op_w
grangertest(rsr, op_w, order = 3)

# H0: op_w does not Granger cause rsr
grangertest(op_w, rsr, order = 3)

