###############################################################################
### Forecasting Korean Inflation, BOK research project by Heejoono Han      ###
###############################################################################

####################
### 2. Forecasting Part
#################### 


setwd("D:/0교재 집필/R code/Ch7 US inflation")    
dir()


FREDMD <- read_excel("FREDMDdata.xlsx",sheet=1)


Y = as.matrix(FREDMD)
mode(Y)

# Loading Data

#main1 파일에서 Y load할 것

# window size, number of forecasts 
#window_size = 252*2
#npred=nrow(Y) - window_size  # npred = 106

npred=132

## Random Walk Model ##
source("functions/func-rw.R")

rw1=rw.rolling.window(Y,npred,1)
rw2=rw.rolling.window(Y,npred,2)
rw3=rw.rolling.window(Y,npred,3)

## AR(4) Model ##
source("functions/func-ar.R")

ar1=ar.rolling.window(Y,npred,1,type="fixed")
ar2=ar.rolling.window(Y,npred,2,type="fixed")
ar3=ar.rolling.window(Y,npred,3,type="fixed")

## Ridge Regression, cross-validation 사용 ##
library(glmnet)
source("functions/func-lassoCV.R")

alpha=0

ridge1=lasso.rolling.window(Y,npred,1,alpha,type="lasso")
ridge2=lasso.rolling.window(Y,npred,2,alpha,type="lasso")
ridge3=lasso.rolling.window(Y,npred,3,alpha,type="lasso")

## LASSO ##
alpha=1

lasso1=lasso.rolling.window(Y,npred,1,alpha,type="lasso")
lasso2=lasso.rolling.window(Y,npred,2,alpha,type="lasso")
lasso3=lasso.rolling.window(Y,npred,3,alpha,type="lasso")

## Adaptive LASSO ##
adalasso1=lasso.rolling.window(Y,npred,1,alpha,type="adalasso")
adalasso2=lasso.rolling.window(Y,npred,2,alpha,type="adalasso")
adalasso3=lasso.rolling.window(Y,npred,3,alpha,type="adalasso")

## Elastic Net ##
alpha=0.5

elasticnet1=lasso.rolling.window(Y,npred,1,alpha,type="lasso")
elasticnet2=lasso.rolling.window(Y,npred,2,alpha,type="lasso")
elasticnet3=lasso.rolling.window(Y,npred,3,alpha,type="lasso")

## Adaptive Elastic Net ##

adaelasticnet1=lasso.rolling.window(Y,npred,1,alpha,type="adalasso")
adaelasticnet2=lasso.rolling.window(Y,npred,2,alpha,type="adalasso")
adaelasticnet3=lasso.rolling.window(Y,npred,3,alpha,type="adalasso")

## Complete Subset Regression (CSR) ##
source("functions/func-csr.R")

csr1=csr.rolling.window(Y,npred,1)
csr2=csr.rolling.window(Y,npred,2)
csr3=csr.rolling.window(Y,npred,3)


## Target Factors ##
source("functions/func-fact.R")
source("functions/func-tfact.R")
source("functions/func-baggit.R")

tfact1=tfact.rolling.window(Y,npred,1)
tfact2=tfact.rolling.window(Y,npred,2)
tfact3=tfact.rolling.window(Y,npred,3)


## Random Forest (RF) ##
source("functions/func-rf.R")
library(randomForest)

rf1=rf.rolling.window(Y,npred,1)
rf2=rf.rolling.window(Y,npred,2)
rf3=rf.rolling.window(Y,npred,3)

## XGBoost ##
# Import the Boosting function
source('functions/func-xgb.R')
library(xgboost)

xgb1=xgb.rolling.window(Y,npred,1)
xgb2=xgb.rolling.window(Y,npred,2)
xgb3=xgb.rolling.window(Y,npred,3)

## Neural Networks(Deep Learning) ##
source("functions/func-nn.R")

library(dplyr)
library(keras)
library(h2o)
h2o.init()

nn1=nn.rolling.window(Y,npred,1)
nn2=nn.rolling.window(Y,npred,2)
nn3=nn.rolling.window(Y,npred,3)


## LSTM ##
library(tensorflow)
library(keras)
library(reticulate)
library(tidyverse)
conda_list()

conda_list()[[1]][4] %>% 
  use_condaenv(required = TRUE)

# tf$constant("Hello TensorFlow!")

# Normalization
normalize <- function(x) {
  return((x-min(x))/(max(x)-min(x)))
}
# Inverse Normalization 
denormalize <- function(x, minval, maxval) {
  x*(maxval-minval) + minval
}

source("functions/func-lstm.R")

lstm1 <- mul.lstm.rolling.window(Y,npred,1)  
lstm2 <- mul.lstm.rolling.window(Y,npred,2)  
lstm3 <- mul.lstm.rolling.window(Y,npred,3)  


# saving entire worksapce
save.image("results_forecasts_US.RData")    


