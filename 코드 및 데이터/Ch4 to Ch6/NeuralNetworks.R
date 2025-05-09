
#install.packages('MASS')
library(MASS)

data(Boston, package="MASS")
set.seed(20)
train = sample(1:nrow(Boston), nrow(Boston)/2)

# Min-max normalization
normalize <- function(x, minval, maxval) {
  (x - minval) / (maxval - minval)
}
# Inverse Normalization 
denormalize <- function(x, minval, maxval) {
  x*(maxval-minval) + minval
}

# train data 기준으로 각 변수의 min/max 저장
x_min <- apply(Boston[train,], 2, min)
x_max <- apply(Boston[train,], 2, max)

# 정규화(모든 변수에 대해 train data 기준으로)
train.norm <- as.data.frame(mapply(normalize,Boston[train,],x_min,x_max))
test.norm <- as.data.frame(mapply(normalize,Boston[-train,],x_min,x_max))

## Single-hidden layer neural network
install.packages('nnet')
library(nnet)
set.seed(30)
single.nn = nnet(medv~.,data=train.norm,size=10,linout=TRUE,decay=0.1,
                maxit=1000,trace=FALSE)

# size=10: single hidden layer with 10 units
# if output is continuous, linout=TRUE.
# decay: regularization (weight decay)
# sigmoid activation function

summary(single.nn)

yhat.norm = predict(single.nn,newdata=test.norm)
yhat.single = denormalize(yhat.norm, x_min["medv"], x_max["medv"]) 
boston.test = Boston[-train,"medv"]
mean((yhat.single-boston.test)^2)




## Multi-hidden layers neural network
install.packages('neuralnet')
library(neuralnet)
set.seed(30)
multi.nn = neuralnet(medv~.,data=train.norm,hidden=c(5,3),
                     act.fct = "logistic",linear.output=T)


# hidden=c(5,3): 첫번째 hidden layer unit이 5, 두번째 hidden layer unit이 3
# act.fct="logistic": sigmoid; "tanh" 가능 
# linear.output=T: Regression; =F: Classification.

# Result
plot(multi.nn)

# Weights
multi.nn$weights

yhat.norm_multi = predict(multi.nn,newdata=test.norm)
yhat.multi = denormalize(yhat.norm_multi, x_min["medv"], x_max["medv"]) 
#boston.test = Boston[-train,"medv"]
mean((yhat.multi-boston.test)^2)


## Neural network with 3 hidden layers 
set.seed(30)
multi.nn3 = neuralnet(medv~.,data=train.norm,hidden=c(7,5,3),
                     act.fct = "logistic",linear.output=T)

plot(multi.nn3)
multi.nn3$weights

yhat.norm_multi3 = predict(multi.nn3,newdata=test.norm)
yhat.multi3 = denormalize(yhat.norm_multi3, x_min["medv"], x_max["medv"]) 
#boston.test = Boston[-train,"medv"]
mean((yhat.multi3-boston.test)^2)
