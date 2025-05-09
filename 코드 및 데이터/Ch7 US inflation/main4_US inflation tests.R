###################
### 4. Testing Part
###################

rm(list=ls())
dir()

load("results_forecasts_US.RData") 

source("functions/gwtest.R")
library(sandwich)

real=tail(Y[,1],npred)

model_names <- c('rw','ar','ridge','lasso','adalasso','elasticnet',
                 'adaelasticnet','csr','tfact','rf','xgb','nn','lstm')
horizons <- c(1, 2, 3)

make_pred_matrix <- function(model_list, npred) {
  mat <- matrix(NA, npred, 3)
  mat[,1] <- model_list[[1]]$pred
  mat[-(1:1),2] <- model_list[[2]]$pred
  mat[-(1:2),3] <- model_list[[3]]$pred
  return(mat)
}

rw_pred  <- make_pred_matrix(list(rw1, rw2, rw3), npred)
ar_pred  <- make_pred_matrix(list(ar1, ar2, ar3), npred)
ridge_pred <- make_pred_matrix(list(ridge1, ridge2, ridge3), npred)
lasso_pred <- make_pred_matrix(list(lasso1, lasso2, lasso3), npred)
adalasso_pred <- make_pred_matrix(list(adalasso1, adalasso2, adalasso3), npred)
elasticnet_pred <- make_pred_matrix(list(elasticnet1, elasticnet2, elasticnet3)
                                    , npred)
adaelasticnet_pred <- make_pred_matrix(list(adaelasticnet1, adaelasticnet2,
                                            adaelasticnet3), npred)
csr_pred  <- make_pred_matrix(list(csr1, csr2, csr3), npred)
tfact_pred  <- make_pred_matrix(list(tfact1, tfact2, tfact3), npred)
rf_pred <- make_pred_matrix(list(rf1, rf2, rf3), npred)
xgb_pred <- make_pred_matrix(list(xgb1, xgb2, xgb3), npred)
nn_pred <- make_pred_matrix(list(nn1, nn2, nn3), npred)
lstm_pred <- make_pred_matrix(list(lstm1, lstm2, lstm3), npred)


run_gw_test <- function(model_pred, benchmark_pred, real, npred, horizons) {
  stat <- pval <- rep(NA, 3)
  for (h in horizons) {
    if (h == 1) {
      gw <- gw.test(model_pred[,h], benchmark_pred[,h], real, tau=h,
                    T=npred, method="NeweyWest")
    } else {
      gw <- gw.test(model_pred[-(1:(h-1)),h], benchmark_pred[-(1:(h-1)),h],
                    real[-(1:(h-1))], tau=h, T=(npred-h+1), method="NeweyWest")
    }
    stat[h] <- gw$statistic
    pval[h] <- gw$p.value
  }
  return(list(stat = stat, pval = pval))
}


results <- list(
  ar     = run_gw_test(ar_pred, rw_pred, real, npred, horizons),
  ridge  = run_gw_test(ridge_pred, rw_pred, real, npred, horizons),
  lasso  = run_gw_test(lasso_pred, rw_pred, real, npred, horizons),
  adalasso = run_gw_test(adalasso_pred, rw_pred, real, npred, horizons),
  elasticnet = run_gw_test(elasticnet_pred, rw_pred, real, npred, horizons),
  adaelasticnet = run_gw_test(adaelasticnet_pred, rw_pred, real, npred,
                              horizons),
  csr     = run_gw_test(csr_pred, rw_pred, real, npred, horizons),
  tfact     = run_gw_test(tfact_pred, rw_pred, real, npred, horizons),
  rf     = run_gw_test(rf_pred, rw_pred, real, npred, horizons),
  xgb     = run_gw_test(xgb_pred, rw_pred, real, npred, horizons),
  nn     = run_gw_test(nn_pred, rw_pred, real, npred, horizons),
  lstm     = run_gw_test(lstm_pred, rw_pred, real, npred, horizons)
)

gw_stat <- do.call(rbind, lapply(results, `[[`, "stat"))
gw_pval <- do.call(rbind, lapply(results, `[[`, "pval"))

rownames(gw_stat) <- rownames(gw_pval) <- names(results)

gw_stat <- round(gw_stat, 2)
gw_pval <- round(gw_pval, 3)

colnames(gw_stat) <- colnames(gw_pval) <- paste0("h=", horizons)

sheets <- list("gw_stat" = gw_stat, "gw_mae_pvalue" = gw_pval)
write.xlsx(sheets, file = "gwtest_US.xlsx", rowNames = TRUE)
####


## Model Confidence Set (MCS) test
library(MCS)

for(i in horizons){
  cat("horizon =", i, "\n")
  
  Pred <- cbind(rw_pred[, i], ar_pred[, i], ridge_pred[, i], lasso_pred[, i],
                adalasso_pred[, i], elasticnet_pred[, i],
                adaelasticnet_pred[, i], csr_pred[, i], tfact_pred[, i],      
                rf_pred[, i], xgb_pred[, i], nn_pred[, i], lstm_pred[, i])
  
  colnames(Pred) <- model_names  # 모형 이름 지정
  
  if(i == 1){
    LOSS <- Pred - real
  } else {
    LOSS <- Pred[-(1:(i - 1)), ] - real[-(1:(i - 1))]
  }
  
  LOSS1 <- LOSS^2       # squared error
  LOSS2 <- abs(LOSS)    # absolute error
  
  # MCS 수행
  SSM_i <- MCSprocedure(LOSS1, alpha = 0.50, B = 5000, statistic = "TR")
  
  print(SSM_i)
}


for(i in horizons){
  cat("horizon =", i, "\n")
  
  Pred <- cbind(rw_pred[, i], ar_pred[, i], ridge_pred[, i], lasso_pred[, i],
                adalasso_pred[, i], elasticnet_pred[, i],
                adaelasticnet_pred[, i], csr_pred[, i], tfact_pred[, i],      
                rf_pred[, i], xgb_pred[, i], nn_pred[, i], lstm_pred[, i])
  
  colnames(Pred) <- model_names  # 모형 이름 지정
  
  if(i == 1){
    LOSS <- Pred - real
  } else {
    LOSS <- Pred[-(1:(i - 1)), ] - real[-(1:(i - 1))]
  }
  
  LOSS1 <- LOSS^2       # squared error
  LOSS2 <- abs(LOSS)    # absolute error
  
  # MCS 수행
  SSM_i <- MCSprocedure(LOSS2, alpha = 0.20, B = 5000, statistic = "TR")
  
  print(SSM_i)
}

# saving entire worksapce
save.image("results_all_US.RData")    

