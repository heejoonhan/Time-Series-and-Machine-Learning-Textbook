
install.packages('ISLR')
library(ISLR)

Hitters = na.omit(Hitters)
n = nrow(Hitters)

# train data set과 test data set으로 구분
set.seed(100)
index = sample(1:n, 163, replace=F)

train = Hitters[index,]
test = Hitters[-index,]

# train data set의 X와 Y 지정 

train.x = model.matrix(Salary~.,data=train)[,-1]
# Salary를 제외한 나머지 변수들을 X로 지정
# 첫 번째 행 절편에 해당하는 1을 제외하고 X를 생성

train.y = train$Salary
# Salary를 종속변수로 지정 

# test data set의 X와 Y 지정
test.y = test$Salary
test.x = model.matrix(Salary~.,data=test)[,-1]


# For ridge and lasso, use 'glmnet' package.
#install.packages('glmnet')
#library(glmnet)

# if alpha=0 -> ridge; if alpha=1 -> lasso.
# if 0<alpha<1 -> elastic net
# Default: standardize = TRUE, The coefficients are always returned on the original scale. Default is standardize=TRUE. If variables are in the same units already, you might not wish to standardize. 


##ridge regression
ridge=glmnet(train.x,train.y,alpha=0,lambda=1000)

coef(ridge)
test.yhat = predict(ridge,newx=test.x)
testmse = mean((test.y - test.yhat)^2)
testmse

# Find a tuning parameter lambda.
grid = c(10^seq(10,0,length=100),0)

ridge = glmnet(train.x,train.y,alpha=0,lambda=grid)

test.yhat = predict(ridge,s=grid,newx=test.x)

test.mse = function (yhat,y) mean((y - yhat)^2)

mse = apply(test.yhat, 2, test.mse, y=test.y)

plot(1:101,mse,type='l',col='red',lwd=2,xlab='Index of lambda',ylab='Test MSE')

l = which(mse == min(mse))
ridge$lambda[l]
round(coef.ridge[,l],3)


##LASSO
grid = c(10^seq(3,0,length=50),0)

lasso = glmnet(train.x, train.y,alpha=1,lambda=grid)

test.yhat = predict(lasso,s=grid,newx=test.x)

mse = apply(test.yhat, 2, test.mse, y=test.y)

plot(1:51,mse,type='l',col='red',lwd=2,xlab='Index of lambda',ylab='Test MSE')

l = which(mse == min(mse))
lasso$lambda[l]
round(coef.lasso[,l],3)
