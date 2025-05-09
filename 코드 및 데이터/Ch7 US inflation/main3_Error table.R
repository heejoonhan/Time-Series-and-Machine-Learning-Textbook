###################################
### 3. Making forecast error tables
###################################

rm(list=ls())

setwd("D:/0교재 집필/R code/Ch7 US inflation") 

load("results_forecasts_US.RData") 
library(stringr)
library(openxlsx)

stack <- NULL
for (i in 1:2) {
  #1.random walk  
  rw = cbind(rw1$errors[i],rw2$errors[i],rw3$errors[i])
  #2. ar
  ar = cbind(ar1$errors[i],ar2$errors[i],ar3$errors[i])
  #3. ridge regression
  ridge = cbind(ridge1$errors[i],ridge2$errors[i],ridge3$errors[i])
  #4. lasso
  lasso = cbind(lasso1$errors[i],lasso2$errors[i],lasso3$errors[i])
  #5. adaptive lasso
  adalasso = cbind(adalasso1$errors[i],adalasso2$errors[i],adalasso3$errors[i])
  #6. elastic net
  elasticnet = cbind(elasticnet1$errors[i],elasticnet2$errors[i],elasticnet3$errors[i])
  #7. adaptive elastic net 
  adaelasticnet = cbind(adaelasticnet1$errors[i],adaelasticnet2$errors[i],adaelasticnet3$errors[i])
  #8. complete subset regression (CSR)
  csr = cbind(csr1$errors[i],csr2$errors[i],csr3$errors[i])
  #9. tfact
  tfact  = cbind(tfact1$errors[i],tfact2$errors[i],tfact3$errors[i])
  #10. random forest (RF)
  rf = cbind(rf1$errors[i],rf2$errors[i],rf3$errors[i])
  #11. XGBoost
  xgb  = cbind(xgb1$errors[i],xgb2$errors[i], xgb3$errors[i])
  #12. NN
  nn  = cbind(nn1$errors[i],nn2$errors[i], nn3$errors[i])
  #13. LSTM
  lstm  = cbind(lstm1$errors[i],lstm2$errors[i], lstm3$errors[i])

  df <-  rbind(rw, ar, ridge, lasso, adalasso, elasticnet, adaelasticnet,  csr, tfact,  rf, xgb, nn, lstm) %>% as.data.frame()
  
  df = round(df, digit=5)
  
  stack <- rbind(stack, df)
}
horizons <- c(1, 2, 3)
colnames(stack) <- paste0("h=", horizons)

nModel = 13 # number of model
error_rmse <- stack[1:nModel,]
error_mae <- stack[(nModel+1):(nModel*2),]

rownames(error_rmse) <- c('rw', 'ar','ridge', 'lasso', 'adalasso', 'elasticnet', 'adaelasticnet',  'csr', 'tfact', 'rf', 'xgb','nn','lstm')
rownames(error_mae) <- c('rw', 'ar','ridge', 'lasso', 'adalasso', 'elasticnet', 'adaelasticnet',  'csr','tfact',  'rf', 'xgb','nn','lstm')

sheets <- list("error_rmse" = error_rmse, "error_mae" = error_mae)
write.xlsx(sheets, file = "errortable_US.xlsx", rowNames=TRUE)

error_mae
apply(error_mae, 2, which.min)



apply(error_rmse, 2, which.min)


#####








