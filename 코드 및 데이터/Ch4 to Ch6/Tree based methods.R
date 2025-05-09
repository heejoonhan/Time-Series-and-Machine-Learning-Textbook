
## Regression Tree 
install.packages('MASS')
library(MASS)
install.packages('tree')
library(tree)

data(Boston, package="MASS")

set.seed(20)
train = sample(1:nrow(Boston), nrow(Boston)/2)

tree.boston = tree(medv~.,Boston, subset=train)

summary(tree.boston)
par(cex = 1.5)
plot(tree.boston)
text(tree.boston,pretty=0)


## Cost-complexity pruning 

cv.boston = cv.tree(tree.boston)
cv.boston

bk = cv.boston$k[which(cv.boston$dev == min(cv.boston$dev))]
# 최적 alpha

prune.boston = prune.tree(tree.boston, k=bk)
plot(prune.boston)
text(prune.boston,pretty=0)

yhat = predict(tree.boston,newdata=Boston[-train,])
boston.test = Boston[-train,"medv"]
mean((yhat-boston.test)^2)


## Bagging and random forest 
install.packages('randomForest')
library(randomForest)

# Bagging(p = m), mtry: split할 때 모든 설명 변수 사용 
bag.boston = randomForest(medv~.,data=Boston,subset=train, mtry=13, importance=TRUE)
bag.boston

yhat.bag = predict(bag.boston, newdata=Boston[-train,])
mean((yhat.bag-boston.test)^2)

# Variable importance 
importance(bag.boston)
varImpPlot(bag.boston)


# Random forest(p < m), mtry: split할 때 설명 변수 4개만 사용 
rf.boston=randomForest(medv~., data=Boston, subset=train, mtry=4, importance=TRUE)
rf.boston

yhat.rf = predict(rf.boston,newdata=Boston[-train,])
mean((yhat.rf-boston.test)^2)

# Variable importance 
importance(rf.boston)
varImpPlot(rf.boston)

# Partial dependence plot: Marginal effect
install.packages('pdp')
library(pdp)
partialPlot_rm <- partial(rf.boston, pred.var = "rm")
plot(partialPlot_rm, type ='l')
partialPlot_lstat <- partial(rf.boston, pred.var = "lstat")
plot(partialPlot_lstat, type ='l')


## Boosting: gradient boosting machine
install.packages('gbm')
library(gbm)
set.seed(123)
gbm.boston = gbm(medv~., data=Boston[train,], distribution='gaussian', 
                 n.trees=10000, interaction.depth=2, shrinkage=0.01)

# Continuous output: distribution = 'gaussian' 
# Binary output: distribution = 'bernoulli'
# n.trees: # of iterations
# interaction.depth: # of terminal nodes in each tree
# shrinkage: lambda

yhat.gbm = predict(gbm.boston, Boston[-train,], n.trees=10000)
mean((yhat.gbm - Boston[-train,]$medv)^2)

summary(gbm.boston, plotit = TRUE)  # Variable importance 

# Partial dependence plot 
plot(gbm.boston, i='rm')
plot(gbm.boston, i='lstat')


## XGBoost
install.packages('xgboost')
library(xgboost)

train_data <- Boston[train, ]
test_data <- Boston[-train, ]

# matrix 형식으로 변환
x_train <- model.matrix(medv ~ . - 1, data = train_data)
y_train <- train_data$medv

x_test <- model.matrix(medv ~ . - 1, data = test_data)
y_test <- test_data$medv

xgb.boston = xgboost(data = x_train, label = y_train, 
  objective = "reg:squarederror", nrounds = 10000, max_depth = 2,          eta = 0.01, verbose = 0, nthread=10)

# objective = "reg:squarederror"  연속형 변수 예측
# nrounds                         트리 개수 = boosting iteration 수
# max_depth = 2,                  interaction.depth
# eta = 0.01,                     shrinkage parameter, 학습 속도
# verbose = 0                     학습 진행을 감추기
# nthread                         병렬 연산을 위해 사용할 CPU thread 수 

yhat.xgb <- predict(xgb.boston, newdata = x_test)
mean((yhat.xgb - y_test)^2)  

importance_matrix <- xgb.importance(model = xgb.boston)
print(importance_matrix)
xgb.plot.importance(importance_matrix)

# Partial dependence plot
x_train_df <- as.data.frame(x_train)
pdp_rm <- partial(xgb.boston, pred.var = "rm", train = x_train_df)
autoplot(pdp_rm)
pdp_lstat <- partial(xgb.boston, pred.var = "lstat", train = x_train_df)
autoplot(pdp_lstat)

