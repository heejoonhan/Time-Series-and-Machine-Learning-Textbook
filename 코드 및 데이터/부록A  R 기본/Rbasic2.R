##### R basic 2 #####

#####################
## 데이터 형태 
#####################

# Numeric (수치형), Logical (논리형), Character (문자형), Complex (복소수형)

# mode() 또는 class() 데이터 유형 출력
mode(3)
mode(3>4)
mode(TRUE)
mode("퀀트")
mode(3+2i)

# 데이터 유형 검증
# is.numeric(x), is.double(x), is.interger(x), is.logical(x), is.character(x), is.complex(x), is.na(x), is.null(x), is.nan(x), is.finite(x), is.infinite(x), is.matarix(x)
# NA (결측치, Not Available or Missing Value)
# NULL 비어있는 값
# NaN 수학적으로 정의가 불가능한 수 Not a Number
# Inf 무한대 

is.double(2.2)  # 실수형 여부  
is.integer(2.2) # 정수형 여부
is.integer(5)
is.integer(1L)  # R은 특이하게도 L을 붙여야 정수로 인식
is.integer(as.integer(1)) # 정수로 데이터 유형을 바꾼 뒤 확인인

# 데이터 유형 변경
# as.numeric(x), as.double(x), as.integer(x), as.logical(x), as.character(x), as.complex(x), as.matrix(x)

is.numeric(1/3)
as.character(1/3)
a = "A"
as.numeric("A")

b = as.character(2)
b
b = as.numeric(b)
b-5

c = "11"
c-5
as.numeric(c)-5


#################################
## working directory 확인 및 변경 
#################################

## get working directory
getwd()  
dir() 

## set your working directory 1: 직접 입력. 다만 경로는 /로 구분
setwd("C:/Users/Heejoon/Desktop/R_Quant_updated")

## set your working directory 2: rstudioapi 패키지 활용
install.packages("rstudioapi")
library(rstudioapi)
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
dir()


survey <- read.csv("old_survey.csv") # csv 데이터 파일 불러오기

height 
survey$height

attach(survey)   #변수 이름을 직접 사용할 수 있도록 등록함
height
detach(survey)   #등록된 변수 이름을 제거함 
height

attach(survey)   

## ----add_column, ----------------------------------------
height_handspan_ratio = height/handspan
survey = cbind(survey, height_handspan_ratio)

## ----remove_column, -------------------------------------
survey = survey[,-c(7:10)]

## ----head_tail-----------------------------------------------------------
head(survey)
tail(survey)

## ----hist_plain, 히스토그램 ----------------------------------------
hist(handedness)

## ----hist_dressed, --------------------------------------
hist(handedness, xlab = 'Handedness Score',
     main = 'Histogram of Handedness Scores',
     ylab = '# of Students')

## ----hist_breaks, 막대 20개로 조정  ---------------------------------------
hist(handedness, breaks = 20, xlab = 'Handedness Score',
     main = 'Histogram of Handedness Scores')

## ----hist_freq, y축 Density -----------------------------------------
hist(handedness, breaks = 20, freq = FALSE,
     xlab = 'Handedness Score', main = 'Histogram of Handedness')

## ----plot_basic, scatter plot ----------------------------------------
plot(height, handspan)

## ----plot_rev, ------------------------------------------
plot(handspan, height)

## ----plot_ornament, -------------------------------------
plot(height, handspan, xlab = "height (in)", ylab = "handspan (cm)")

## ----plot_col, 색깔 ------------------------------------------
plot(height, handspan, xlab = "height (in)",
     ylab = "handspan (cm)", col = "red")

## ----plot_pch, 형태 ------------------------------------------
plot(height, handspan, xlab = "height (in)",
     ylab = "handspan (cm)", col = "red", pch = 3)

## ----plot_type_l, connected line ---------------------------------------
plot(height, handspan, xlab = "height (in)",
     ylab = "handspan (cm)", col = "red", pch = 3, type = 'l')

## ----pairs---------------------------------------------------------------
A = cbind(handedness, handspan, height)
pairs(A)

## ----boxplot, median, interquartile (25% and 75% quantile), min and max, outlier------------------
boxplot(handspan, ylab = "Handspan(cm)")

## ----boxplot_comparison--------------------------------------------------
boxplot(handspan ~ sex, ylab= "Handspan (cm)", main = "Handspan by Sex")



## ----summary-------------------------------------------------------------
summary(survey)

## ----sum_ missing observation이 있을 때------------------------------------
sum(height)

## ----na.rm: ignore missing observations; NA를 제거하고 sum ----------------
sum(height, na.rm = TRUE)

## ----mean_na.rm_1, mean 평균-----------------------------
mean(height, na.rm = TRUE)

## ----mean_na.rm_2--------------------------------------------------------

mean(1:3)
#  missing observation을 무시하기 때문에, 평균은 (1+2)/2
mean(c(1, 2, NA), na.rm = TRUE)

## ----var_na.rm, variance 분산---------------------------------------------
var(height, na.rm = TRUE)

## ----sd_na.rm, standard deviation 표준편차 -------------------------------
sd(height, na.rm = TRUE)

## ----표준편차는 분산의 제곱근----------------------------------------------
sqrt(var(height, na.rm = TRUE))

## ----median_na.rm, median 중위값 또는 중앙값-------------------------------
median(height, na.rm = TRUE)

## ----quantile_5no, quantile 분위수 --------------------------------------
quantile(height, na.rm = TRUE)

## ----quantile_probs, 분위수------------------------------------------------
quantile(height, na.rm = TRUE, probs = 0.3)

## ----quantile_many_probs-------------------------------------------------
quantile(height, na.rm = TRUE, probs = c(0.1, 0.3, 0.7, 0.9))

## ----iqr, Inter-Quartile Range---------------------------------------------
IQR(height, na.rm = TRUE)

## ----iqr_w_quantile, ------------------------------------------------------
x = quantile(height, na.rm = TRUE, probs = c(.25, .75))
x[2] - x[1]

## ----max_min, 최대 최소----------------------------------------------------
max(height, na.rm = TRUE)
min(height, na.rm = TRUE)

## ----range_by_hand-------------------------------------------------------
max(height, na.rm = TRUE) - min(height, na.rm = TRUE)

## ----range---------------------------------------------------------------
range(height, na.rm = TRUE)

## ----which.max_min-------------------------------------------------------
which.max(height)     # 몇번째 observation? 
height[which.max(height)]
height[which.min(height)]


## ----is_na  NA인지 여부 ---------------------------------------------------
x = c(1, 2, NA, 3, NA, 4)
is.na(x) 

## ----not-----------------------------------------------------------------
!is.na(x)

## ----and-----------------------------------------------------------------
y = c(NA, 1, NA, 2, 3, NA)
is.na(y)
!is.na(y)
!is.na(x) & !is.na(y)   #둘다 NA가 아님 

detach(survey)
