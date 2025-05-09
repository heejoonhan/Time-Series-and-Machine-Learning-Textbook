
## LSTM 실행을 위한 준비

install.packages("remotes")
library(remotes)
library(reticulate)
remotes::install_github("rstudio/tensorflow")
reticulate::install_python()

reticulate::use_python("C:/Users/HERO/anaconda3/python.exe")

###################
library(tensorflow)
library(keras)
library(reticulate)
library(tidyverse)
conda_list()

conda_list()[[1]][4] %>% 
  use_condaenv(required = TRUE)

tf$constant("Hello TensorFlow!")

###################

setwd('D:/0교재 집필/R code/Ch4 example')
dir()

library("readxl")
# load data
FREDMD <- read_excel("FREDMDdata.xlsx",sheet=1)

# Min-max normalization
normalize <- function(x, minval, maxval) {
  (x - minval) / (maxval - minval)
}
# Inverse Normalization 
denormalize <- function(x, minval, maxval) {
  x*(maxval-minval) + minval
}

# 변수의 min/max 저장
x_min <- apply(FREDMD, 2, min)
x_max <- apply(FREDMD, 2, max)

# 정규화
FREDMD.norm <- as.data.frame(mapply(normalize,FREDMD,x_min,x_max))

y <- FREDMD.norm[["INDPRO"]]
xdata <- model.matrix(INDPRO~.,data = FREDMD.norm)[,-1]

# 시점당 변수 수
n_features <- ncol(xdata)
timestep   <- 4  # 과거 4개월을 입력으로 사용
horizon    <- 1  # 예측 대상기간(forecast horizon)

# 입력 데이터를 3D-array로 만드는 함수  
create_dataset <- function(x, y, timestep, horizon) {
  X <- array(NA, dim = c(nrow(x) - timestep - horizon + 1,
                         timestep, ncol(x)))
  Y <- y[(timestep + horizon):length(y)]
  for (i in 1:(nrow(x) - timestep  - horizon + 1)) {
    X[i,,] <- as.matrix(x[i:(i + timestep - 1), ])
  }
  return(list(X = X, y = Y))
}

ds <- create_dataset(xdata, y, timestep, horizon)

X_train <- ds$X
y_train <- ds$y

create_X.out <- function(x, timestep) {
  X_last <- array(NA, dim = c(1, timestep, ncol(x)))
  X_last[1,,] <- as.matrix(tail(x, timestep))
  return(X_last)
}

X.out.arr <- create_X.out(xdata, timestep)


## Set random seeds
set.seed(42)         # R에서의 난수수 제어
set_random_seed(42)  # 딥러닝 내부의 가중치 초기화 제어


## LSTM 모형 설정 및 학습
model <- keras_model_sequential() %>%
  layer_lstm(units = 32, input_shape = c(timestep, n_features)
             ,stateful = FALSE) %>%
  layer_dense(units = 1)


model %>% compile(
  loss = "mse",
  optimizer = "adam"
)

summary(model)

model %>% fit(X_train, y_train, epochs = 100,
  batch_size = 50, shuffle = FALSE, verbose = 2
)


## h-step ahead out-of-sample forecast
h_step_ahead_pred <- model %>% predict(X.out.arr)

y_min <- x_min["INDPRO"]
y_max <- x_max["INDPRO"]
y_pred <- denormalize(h_step_ahead_pred, y_min, y_max)


## 2-layer LSTM model with dropout (rate = 0.3) 

model2 <- keras_model_sequential() %>%
   layer_lstm(units = 32,
                      input_shape = c(timestep, n_features),
                      stateful = FALSE,
                      return_sequences = TRUE) %>% 
   layer_dropout(rate = 0.3) %>% 
   layer_lstm(units = 16) %>% 
   layer_dropout(rate = 0.3) %>% 
   layer_dense(units = 1)

model2 %>% compile(
  loss = "mse",
  optimizer = "adam"
)

summary(model2)

Dmodel2 %>% fit(X_train, y_train, epochs = 100,
              batch_size = 50, shuffle = FALSE, verbose = 2
)

## h-step ahead out-of-sample forecast
h_step_ahead_pred2 <- model2 %>% predict(X.out.arr)
y_pred2 <- denormalize(h_step_ahead_pred2, y_min, y_max)

