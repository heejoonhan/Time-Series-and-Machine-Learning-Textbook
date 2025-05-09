
# 필요한 패키지 로드
library("readxl")
library(Boruta)

# 데이터 불러오기
FREDMD = read_excel("FREDMDdata.xlsx",sheet=1)

# 종속 변수와 설명 변수 설정
y = FREDMD$INDPRO
xdata = model.matrix(INDPRO~.,data = FREDMD)[,-1]

# Boruta 알고리즘 실행
set.seed(42)  
boruta_result <- Boruta(xdata, y, maxRuns = 100)
par(cex = 1.5)
plot(boruta_result)
attstats <- attStats(boruta_result)
head(attstats)

# 중요 변수 선택  
confirmed_vars <- rownames(attstats)[attstats$decision == "Confirmed"]
nImp <- length(confirmed_vars)

order = order(attstats$meanImp, decreasing = T)
Variables_selected = order[1:nImp]
  
# 최종 선택된 변수 이름 (중요도 순, Confirm된 것만)
Variable_names_selected <- rownames(attstats)[order][1:nImp]
