###############################
### 1. Data transformation Part
###############################
rm(list=ls())
setwd("D:/0교재 집필/R code/Ch7 US inflation")  
dir()

# FRED-MD 2016년 1월 vintage 데이터
# 결측치 포함한 데이터 제거, 1959년 11월 ~ 2000년 12월로 제한   
data = read.csv("2016-01.csv")
complete.cases(data)

y1 = diff(log(data$CPIAUCSL))[-(1:2)]
y2 = diff(log(data$PCEPI))[-(1:2)]

data = subset(data, select = -c(CPIAUCSL, PCEPI))
tcode = data[1,]  # 첫 번째 행: Tcode
data = data[-1,]  # Tcode 행 제거 
tdata = data[-(1:2),] # 최초 2 표본 제거(2차 차분 고려)

for (i in 2:ncol(data)){
  if(tcode[i] == 1){
    tdata[,i] <- data[-(1:2),i]
  } # 그대로  
  if(tcode[i] == 2){
    tdata[,i] <- diff(data[-1,i])
  } # 1차 차분
  # tdoce ==3(2차 차분)에 해당하는 데이터는 없음
  if(tcode[i] == 4){
    tdata[,i] <- log(data[-(1:2),i])
  } # log
  if(tcode[i] == 5){
    tdata[,i] <- diff(log(data[-1,i]))
  } # log 취한 뒤 1차 차분
  if(tcode[i] == 6){
    tdata[,i] <- diff(diff(log(data[,i])))
  } # log 취한 뒤 2차 차분
  if(tcode[i] == 7){
    tdata[,i] <- diff(data[-1,i]/data[1:(nrow(data)-1),i])
  } # 증가율의 1차 차분
}

tdata = tdata[,-1] # date 제거, 1960년 1월 ~ 2000년 12월   
Y = cbind(CPI = y1, PCE = y2, tdata) 
Y= as.matrix(Y)

# Medeiros et al.(2021) 데이터 확인 결과 1960년 2월부터 사용
Y = Y[-1,] 

inflation <- Y[,1]
#par(cex.main = 2, cex.lab = 2, cex.axis = 2)
plot(inflation, type='l')

