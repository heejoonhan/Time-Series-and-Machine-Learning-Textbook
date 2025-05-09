getwd()
setwd('D:/0교재 집필/R code/Ch4 example')
dir()

install.packages('readxl')
library("readxl")
install.packages('glmnet')
library(glmnet)

# load data
FREDMD = read_excel("FREDMDdata.xlsx",sheet=1)

y = FREDMD$INDPRO
xdata = model.matrix(INDPRO~.,data = FREDMD)[,-1]
## OLS estimation 
ols = lm(INDPRO~.,data = FREDMD)
summary(ols)

##ridge regression
ridge=cv.glmnet(xdata,y,alpha=0)

ridge$lambda.min

coef(ridge, ridge$lambda.min)

coef(ridge)  # which lambda ?
coef(ridge, ridge$lambda.1se)  

ridge$lambda.1se


##LASSO
lasso = cv.glmnet(xdata,y,alpha=1)

lasso$lambda.min

coef(lasso, lasso$lambda.min)

lasso$lambda.1se

selected=which(coef(lasso, lasso$lambda.min)[-1]!=0)   
#non-zero coefficients excluding intercept

selected

coef(lasso, lasso$lambda.min)[-1][selected]


## Adaptive LASSO
coef.lasso = coef(lasso, lasso$lambda.min)

penalty=(abs(coef.lasso[-1])+1/sqrt(length(y)))^(-1)    
# coef.lasso[-1]: lasso estimates excluding intercept

adlasso=cv.glmnet(xdata,y,penalty.factor = penalty,alpha=1)

selected_ad=which(coef(adlasso, adlasso$lambda.min)[-1]!=0)   

coef(adlasso, adlasso$lambda.min)[-1][selected_ad]


## OLS with selected variables
newx = xdata[,selected_ad]
ols = lm(y~newx)
summary(ols)


## LASSO with information criteria
install.packages('devtools')
# Installing 'HDeconometrics' Package from Github

library(devtools)  
install_github("gabrielrvsc/HDeconometrics")
library(HDeconometrics)

lasso_ic=ic.glmnet(xdata,y,crit = "bic",alpha = 1)

lasso_ic$lambda

selected_ic=which(coef(lasso_ic, lasso_ic$lambda)[-1]!=0)

coef(lasso_ic, lasso_ic$lambda)[-1][selected_ic]


##Elastic Net
elasticnet = cv.glmnet(xdata,y,alpha=0.5)

elasticnet$lambda.min

coef(elasticnet, elasticnet$lambda.min)

selected_en=which(coef(elasticnet, elasticnet$lambda.min)[-1]!=0)   

selected_en

coef(elasticnet, elasticnet$lambda.min)[-1][selected_en]

