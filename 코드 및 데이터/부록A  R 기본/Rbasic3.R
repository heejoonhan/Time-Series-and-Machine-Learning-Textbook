##### R basic 3 #####

#install.packages("rstudioapi")
library(rstudioapi)
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
dir()

survey <- read.csv("old_survey.csv")

attach(survey)   

survey = cbind(handedness, height, handspan)

## ----cor: correlation, use = "complete.obs"은 모든 missing observations 제거 -----------
cor(handspan, height, use = "complete.obs")

## ----cor_dt--------------------------------------------------------------
cor(survey, use = "complete.obs")
#alternatively, there's the na.omit function. na.omit은 NA를 제거 
cor(na.omit(survey))

# same correlation as in line 15?

## ----find_missing--------------------------------------------------------
which(is.na(handedness) & !is.na(height) & !is.na(handspan))

## ----cov: covariance ------------------------------------------------------
cov(survey, use = "complete.obs")
cov(na.omit(survey))


#####################
## function 
#####################

## ----function_write------------------------------------------------------
z.score = function(x){
  z = (x - mean(x, na.rm = TRUE))/sd(x, na.rm = TRUE)
  return(z)
}

## ----mymean--------------------------------------------------------------
mymean = function(x){
  x = x[!is.na(x)]    # x is overwritten
  x.bar = sum(x)/length(x)
  return(x.bar)
}

## ----mean: R에 내장된 함수----------------------------------
mean(height, na.rm = TRUE)

## ----mymean: 위에서 정의된 함수-----------------------------------------
mymean(height)

## ----mymean2-------------------------------------------------------------
mymean2 = function(x){
  x.bar = sum(x, na.rm = TRUE)/length(x)
  return(x.bar)
}
mymean2(height)
# mymean과 mymean2의 비교 
# mymean2 함수 안에서 sum ignores missing observations but length does not.
 

## ----myvar---------------------------------------------------------------
myvar = function(x){
  x = x[!is.na(x)]
  s.squared = sum((x-mymean(x))^2)/(length(x) - 1)
  return(s.squared)
}

## ----myvar_test----------------------------------------------------------
var(handspan, na.rm = TRUE)
myvar(handspan)

## ----exercise_2----------------------------------------------------------
#Exercise #2 - Write a Function to Calculate Skewness
skew = function(x){
  x = x[!is.na(x)]
  numerator = sum((x - mean(x))^3)/length(x)
  denominator = sd(x)^3
  return(numerator/denominator)  
}
skew(handedness)

#install.packages("data.table") # 패키지 설치
library(data.table) # 패키지 로딩

## ----function_return_data.table------------------------------------------
summary.stats = function(x){
  x = x[!is.na(x)]
  sample.mean = mean(x)
  std.dev  = sd(x)
  out = data.table(sample.mean, std.dev)
  return(out)
}
results = summary.stats(handedness)
results
results$sample.mean
results$std.dev

## ----mycov---------------------------------------------------------------
mycov = function(x, y){
  
  keep = !is.na(x) & !is.na(y)
  x = x[keep]
  y = y[keep]
  
  n = length(x)
  
  s.xy = sum( (x - mean(x)) * (y - mean(y)) ) / (n-1)
  return(s.xy)
}

## ----mycov 함수와 R에 내장된 함수 비교-----------------------------------------
cov(handspan, handedness, use = "complete.obs")
mycov(handspan, handedness)

detach(survey)
