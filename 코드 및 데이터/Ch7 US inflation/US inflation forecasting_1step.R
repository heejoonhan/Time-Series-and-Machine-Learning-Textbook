
# 예측 대상기간: 1개월 앞 예측

## Random Walk Model ##
source("functions/func-rw.R")
rw1=rw.rolling.window(Y,npred,1)

## AR(4) Model ##
source("functions/func-ar.R")
ar1=ar.rolling.window(Y,npred,1,type="fixed")

## Ridge Regression, cross-validation 사용 ##
library(glmnet)
source("functions/func-lassoCV.R")
alpha=0
ridge1=lasso.rolling.window(Y,npred,1,alpha,type="lasso")

## LASSO ##
alpha=1
lasso1=lasso.rolling.window(Y,npred,1,alpha,type="lasso")

## Adaptive LASSO ##
adalasso1=lasso.rolling.window(Y,npred,1,alpha,type="adalasso")

## Elastic Net ##
alpha=0.5
elasticnet1=lasso.rolling.window(Y,npred,1,alpha,type="lasso")

## Adaptive Elastic Net ##
adaelasticnet1=lasso.rolling.window(Y,npred,1,alpha,type="adalasso")

## Complete Subset Regression (CSR) ##
source("functions/func-csr.R")
csr1=csr.rolling.window(Y,npred,1)

## Target Factors ##
source("functions/func-fact.R")
source("functions/func-tfact.R")
source("functions/func-baggit.R")
tfact1=tfact.rolling.window(Y,npred,1)

## Random Forest (RF) ##
source("functions/func-rf.R")
library(randomForest)
rf1=rf.rolling.window(Y,npred,1)

## XGBoost ##
source('functions/func-xgb.R')
library(xgboost)
xgb1=xgb.rolling.window(Y,npred,1)

## Neural Networks(Deep Learning) ##
source("functions/func-nn.R")
library(dplyr)
library(keras)
library(h2o)
h2o.init()

nn1=nn.rolling.window(Y,npred,1)

## LSTM ##
library(tensorflow)
library(keras)
library(reticulate)
library(tidyverse)
conda_list()

conda_list()[[1]][4] %>% 
  use_condaenv(required = TRUE)

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

# saving entire worksapce
save.image("results_forecasts_US.RData") 


